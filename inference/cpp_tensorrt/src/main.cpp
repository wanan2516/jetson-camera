#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "opencv2/opencv.hpp"

#include "inference.hpp"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {

/**
 * 从输入流中精确读取 size 个字节
 */
bool ReadExact(std::istream& in, char* data, std::size_t size) {
    std::size_t total = 0;

    while (total < size) {
        in.read(data + total, static_cast<std::streamsize>(size - total));

        std::streamsize got = in.gcount();

        if (got <= 0) {
            return false;
        }

        total += static_cast<std::size_t>(got);
    }

    return true;
}

/**
 * 向输出流中精确写入 size 个字节
 */
bool WriteExact(std::ostream& out, const char* data, std::size_t size) {
    out.write(data, static_cast<std::streamsize>(size));
    return static_cast<bool>(out);
}

/**
 * 从输入流中读取 uint32 小端整数
 *
 * Python 端对应:
 * struct.pack("<I", size)
 */
bool ReadUInt32LE(std::istream& in, uint32_t& value) {
    unsigned char buf[4];

    if (!ReadExact(in, reinterpret_cast<char*>(buf), 4)) {
        return false;
    }

    value =
        static_cast<uint32_t>(buf[0]) |
        (static_cast<uint32_t>(buf[1]) << 8) |
        (static_cast<uint32_t>(buf[2]) << 16) |
        (static_cast<uint32_t>(buf[3]) << 24);

    return true;
}

/**
 * 向输出流中写入 uint32 小端整数
 *
 * Python 端对应:
 * struct.unpack("<I", header)[0]
 */
bool WriteUInt32LE(std::ostream& out, uint32_t value) {
    unsigned char buf[4];

    buf[0] = static_cast<unsigned char>(value & 0xFF);
    buf[1] = static_cast<unsigned char>((value >> 8) & 0xFF);
    buf[2] = static_cast<unsigned char>((value >> 16) & 0xFF);
    buf[3] = static_cast<unsigned char>((value >> 24) & 0xFF);

    return WriteExact(out, reinterpret_cast<const char*>(buf), 4);
}

/**
 * C++ -> Python 输出协议:
 *   4 bytes: alarm_flag，uint32，小端，0/1
 *   4 bytes: JPEG 结果图长度，uint32，小端
 *   N bytes: JPEG 编码后的结果图
 */
bool WriteResultPacket(
    std::ostream& out,
    bool alarm_flag,
    const std::vector<unsigned char>& output_buffer
) {
    if (output_buffer.size() > UINT32_MAX) {
        return false;
    }

    const uint32_t alarm_value = alarm_flag ? 1U : 0U;
    const uint32_t output_size = static_cast<uint32_t>(output_buffer.size());

    if (!WriteUInt32LE(out, alarm_value)) {
        return false;
    }

    if (!WriteUInt32LE(out, output_size)) {
        return false;
    }

    if (output_size == 0) {
        return true;
    }

    return WriteExact(
        out,
        reinterpret_cast<const char*>(output_buffer.data()),
        output_buffer.size()
    );
}

bool WriteEmptyResultPacket(std::ostream& out) {
    const std::vector<unsigned char> empty;
    return WriteResultPacket(out, false, empty);
}

bool StartsWithDashDash(const std::string& s) {
    return s.rfind("--", 0) == 0;
}

}  // namespace


int main(int argc, char** argv) {
#ifdef _WIN32
    // Windows 下需要设置 stdin/stdout 为二进制模式
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif

    if (argc < 3) {
        std::cerr << "Usage: ./camera_tensorrt_server "
                  << "[engine_path] [config_json] "
                  << "[roi_config_json optional] "
                  << "[--prestart] [--settle]"
                  << std::endl;
        return -1;
    }

    const std::string engine_path = argv[1];
    const std::string config_path = argv[2];

    /**
     * roi_config_path 可选。
     *
     * 如果只传:
     *   ./camera_tensorrt_server engine config.json
     *
     * 则 roi_config_path 默认等于 config_path。
     *
     * 如果传:
     *   ./camera_tensorrt_server engine config.json roi_config.json
     *
     * 则使用第三个参数作为 ROI fallback 配置。
     */
    std::string roi_config_path = config_path;
    int flag_start_index = 3;

    if (argc >= 4) {
        const std::string maybe_roi_path = argv[3];

        if (!StartsWithDashDash(maybe_roi_path)) {
            roi_config_path = maybe_roi_path;
            flag_start_index = 4;
        }
    }

    bool prestart_mode = false;
    bool settle_single_frame = false;

    for (int i = flag_start_index; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--prestart") {
            prestart_mode = true;
        } else if (arg == "--settle") {
            settle_single_frame = true;
        } else {
            std::cerr << "[C++] warning: unknown argument ignored: "
                      << arg << std::endl;
        }
    }

    try {
        /**
         * 这里会加载 TensorRT engine、配置文件、ROI 配置。
         * 只执行一次，不会每一帧重新加载。
         *
         * 因此前端修改配置后，需要 Python 端重启 C++ 子进程。
         */
        CameraTensorRTInfer infer(
            engine_path,
            config_path,
            roi_config_path,
            prestart_mode,
            settle_single_frame
        );

        std::cerr << "[C++] TensorRT inference server started." << std::endl;
        std::cerr << "[C++] engine_path: " << engine_path << std::endl;
        std::cerr << "[C++] config_path: " << config_path << std::endl;
        std::cerr << "[C++] roi_config_path: " << roi_config_path << std::endl;
        std::cerr << "[C++] prestart_mode: "
                  << (prestart_mode ? "true" : "false") << std::endl;
        std::cerr << "[C++] settle_single_frame: "
                  << (settle_single_frame ? "true" : "false") << std::endl;

        while (true) {
            uint32_t input_size = 0;

            /**
             * 通信协议:
             *
             * Python -> C++:
             *   4 bytes: JPEG 数据长度，uint32，小端
             *   N bytes: JPEG 编码后的图像
             *
             * C++ -> Python:
             *   4 bytes: alarm_flag，uint32，小端，0/1
             *   4 bytes: JPEG 结果图长度，uint32，小端
             *   N bytes: JPEG 编码后的结果图
             */

            if (!ReadUInt32LE(std::cin, input_size)) {
                std::cerr << "[C++] stdin closed, server exit." << std::endl;
                break;
            }

            /**
             * Python 发送长度 0，表示让 C++ 正常退出
             */
            if (input_size == 0) {
                std::cerr << "[C++] receive stop signal, server exit." << std::endl;
                break;
            }

            /**
             * 简单限制一下输入包大小，防止异常数据导致内存爆掉
             */
            if (input_size > 50 * 1024 * 1024) {
                std::cerr << "[C++] input packet too large: "
                          << input_size << " bytes" << std::endl;

                WriteEmptyResultPacket(std::cout);
                std::cout.flush();
                continue;
            }

            std::vector<unsigned char> input_buffer(input_size);

            if (!ReadExact(
                    std::cin,
                    reinterpret_cast<char*>(input_buffer.data()),
                    input_buffer.size())) {
                std::cerr << "[C++] failed to read input image bytes." << std::endl;
                break;
            }

            cv::Mat input_img = cv::imdecode(input_buffer, cv::IMREAD_COLOR);

            if (input_img.empty()) {
                std::cerr << "[C++] failed to decode input image." << std::endl;

                WriteEmptyResultPacket(std::cout);
                std::cout.flush();
                continue;
            }

            CameraInferResult infer_result;

            try {
                /**
                 * 核心推理:
                 * 输入 cv::Mat
                 * 输出已经画好 ROI、检测框、FPS、报警状态的 cv::Mat
                 */
                infer_result = infer.Infer(input_img);
            } catch (const std::exception& e) {
                std::cerr << "[C++] inference failed: "
                          << e.what() << std::endl;

                WriteEmptyResultPacket(std::cout);
                std::cout.flush();
                continue;
            }

            std::vector<unsigned char> output_buffer;

            std::vector<int> encode_params = {
                cv::IMWRITE_JPEG_QUALITY,
                95
            };

            bool encode_ok = cv::imencode(
                ".jpg",
                infer_result.image,
                output_buffer,
                encode_params
            );

            if (!encode_ok || output_buffer.empty()) {
                std::cerr << "[C++] failed to encode output image." << std::endl;

                WriteEmptyResultPacket(std::cout);
                std::cout.flush();
                continue;
            }

            if (output_buffer.size() > UINT32_MAX) {
                std::cerr << "[C++] output image too large." << std::endl;

                WriteEmptyResultPacket(std::cout);
                std::cout.flush();
                continue;
            }

            if (!WriteResultPacket(
                    std::cout,
                    infer_result.alarm,
                    output_buffer)) {
                std::cerr << "[C++] failed to write output packet." << std::endl;
                break;
            }

            std::cout.flush();
        }

    } catch (const std::exception& e) {
        std::cerr << "[C++] fatal error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}