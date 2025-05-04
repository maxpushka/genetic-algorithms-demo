#include "include/logging.h"

#include <filesystem>

#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

std::shared_ptr<spdlog::logger> Logger::s_main_logger;
std::shared_ptr<spdlog::logger> Logger::s_file_logger;
std::shared_ptr<spdlog::logger> Logger::s_results_logger;

void Logger::init(const std::string& log_file, spdlog::level::level_enum level) {
    // Create logs directory if it doesn't exist
    std::filesystem::create_directories("logs");

    // Create a console sink
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
    
    // Create a file sink
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("logs/" + log_file, true);
    file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] %v");
    
    // Create a rotating file sink for results
    auto results_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
        "logs/results.log", 10 * 1024 * 1024, 3);
    results_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] %v");
    
    // Create loggers with both sinks
    s_main_logger = std::make_shared<spdlog::logger>("main", spdlog::sinks_init_list{console_sink, file_sink});
    s_file_logger = std::make_shared<spdlog::logger>("file", file_sink);
    s_results_logger = std::make_shared<spdlog::logger>("results", results_sink);
    
    // Set global log level
    set_level(level);
    
    // Register loggers
    spdlog::register_logger(s_main_logger);
    spdlog::register_logger(s_file_logger);
    spdlog::register_logger(s_results_logger);
    
    // Set as default logger
    spdlog::set_default_logger(s_main_logger);
    
    s_main_logger->info("Logger initialized");
}

std::shared_ptr<spdlog::logger>& Logger::get_main_logger() {
    return s_main_logger;
}

std::shared_ptr<spdlog::logger>& Logger::get_file_logger() {
    return s_file_logger;
}

std::shared_ptr<spdlog::logger>& Logger::get_results_logger() {
    return s_results_logger;
}

void Logger::set_level(spdlog::level::level_enum level) {
    spdlog::set_level(level);
    
    if (s_main_logger) {
        s_main_logger->set_level(level);
    }
    
    if (s_file_logger) {
        s_file_logger->set_level(level);
    }
    
    if (s_results_logger) {
        s_results_logger->set_level(level);
    }
}