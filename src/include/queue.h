#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

// Thread-safe queue for asynchronous I/O operations
template <typename T>
class ThreadSafeQueue {
 private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  std::condition_variable cond_;
  bool done_ = false;

 public:
  ThreadSafeQueue() = default;

  void push(T item) {
    std::lock_guard lock(mutex_);
    queue_.push(
        std::move(item));  // Using push with move since we have a complete item
    cond_.notify_one();
  }

  // Variadic template for emplacing items directly
  template <typename... Args>
  void emplace(Args &&...args) {
    std::lock_guard lock(mutex_);
    queue_.emplace(std::forward<Args>(args)...);  // Construct in-place
    cond_.notify_one();
  }

  bool try_pop(T &item) {
    std::lock_guard lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    item = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  std::shared_ptr<T> try_pop() {
    std::lock_guard lock(mutex_);
    if (queue_.empty()) {
      return nullptr;
    }
    std::shared_ptr<T> res(std::make_shared<T>(std::move(queue_.front())));
    queue_.pop();
    return res;
  }

  void wait_and_pop(T &item) {
    std::unique_lock lock(mutex_);
    cond_.wait(lock, [this]() { return !queue_.empty() || done_; });
    if (done_ && queue_.empty()) {
      return;
    }
    item = std::move(queue_.front());
    queue_.pop();
  }

  std::shared_ptr<T> wait_and_pop() {
    std::unique_lock lock(mutex_);
    cond_.wait(lock, [this]() { return !queue_.empty() || done_; });
    if (done_ && queue_.empty()) {
      return nullptr;
    }
    std::shared_ptr<T> res(std::make_shared<T>(std::move(queue_.front())));
    queue_.pop();
    return res;
  }

  bool empty() const {
    std::lock_guard lock(mutex_);
    return queue_.empty();
  }

  size_t size() const {
    std::lock_guard lock(mutex_);
    return queue_.size();
  }

  void done() {
    std::lock_guard lock(mutex_);
    done_ = true;
    cond_.notify_all();
  }

  bool is_done() const {
    std::lock_guard lock(mutex_);
    return done_;
  }
};