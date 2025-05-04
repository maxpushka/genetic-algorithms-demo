#pragma once

#include <memory>
#include <string>

#include "spdlog/spdlog.h"

// Logging configuration for the application
class Logger {
 public:
  // Initialize the logging system
  static void init(const std::string& log_file = "ga_log.txt",
                   spdlog::level::level_enum level = spdlog::level::info);

  // Get the main logger
  static std::shared_ptr<spdlog::logger>& get_main_logger();

  // Get the file logger
  static std::shared_ptr<spdlog::logger>& get_file_logger();

  // Get the results logger (for statistics output)
  static std::shared_ptr<spdlog::logger>& get_results_logger();

  // Set global log level
  static void set_level(spdlog::level::level_enum level);

 private:
  static std::shared_ptr<spdlog::logger> s_main_logger;
  static std::shared_ptr<spdlog::logger> s_file_logger;
  static std::shared_ptr<spdlog::logger> s_results_logger;
};

// Convenient macros for logging
#define LOG_TRACE(...) ::Logger::get_main_logger()->trace(__VA_ARGS__)
#define LOG_DEBUG(...) ::Logger::get_main_logger()->debug(__VA_ARGS__)
#define LOG_INFO(...) ::Logger::get_main_logger()->info(__VA_ARGS__)
#define LOG_WARN(...) ::Logger::get_main_logger()->warn(__VA_ARGS__)
#define LOG_ERROR(...) ::Logger::get_main_logger()->error(__VA_ARGS__)
#define LOG_CRITICAL(...) ::Logger::get_main_logger()->critical(__VA_ARGS__)

// Macros for file logging
#define FILE_LOG_INFO(...) ::Logger::get_file_logger()->info(__VA_ARGS__)
#define FILE_LOG_ERROR(...) ::Logger::get_file_logger()->error(__VA_ARGS__)

// Macros for results logging
#define RESULT_LOG(...) ::Logger::get_results_logger()->info(__VA_ARGS__)