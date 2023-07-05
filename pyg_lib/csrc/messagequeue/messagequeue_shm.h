#include <semaphore.h>
#include <stdio.h>
#include <string.h>
#include <sys/shm.h>
#include <torch/extension.h>
#include <atomic>
#include <iostream>
#include <thread>
#include <vector>

#ifndef MESSAGEQUEUE_SHM_H
#define MESSAGEQUEUE_SHM_H

#define MAX_SAMPLE_NUM 32768

namespace pyg_distributed {
class SHMMessageQueue {
 private:
  // We have to use the relative address here as different process have
  // different address spaces, when attaching the shared memory to the running
  // process The relative is calculated by: (current process addr - head_addr)
  std::size_t shm_end_addr_relative;
  std::atomic<size_t> reader_addr_relative;
  std::atomic<size_t> writer_addr_relative;
  std::atomic<size_t> address_book_write_idx;
  std::atomic<size_t> address_book_read_idx;
  std::atomic<size_t> num_of_samples;
  size_t address_book_arr[MAX_SAMPLE_NUM] = {};
  size_t size_of_shm;

  // Sempahores for controlling reading/writing of the shared memory spaces
  sem_t write_sem;
  sem_t address_book_sem;
  sem_t read_sem;

  // Functions to serialize and deserialize the data from the buffer
  char* SerializeATensor(void* shm_ptr, const torch::Tensor& tensor);
  torch::Tensor DeserializeATensor(void* shm_ptr);
  std::string DeserializeAString(void* shm_ptr);
  char* SerializeAString(void* shm_ptr, const std::string& str);
  char* SerializeADictOfTensors(
      void* shm_ptr,
      const std::map<std::string, torch::Tensor>& dict_of_tensors);
  py::dict DeSerializeADictOfTensors(void* shm_ptr);

  // Helper functions to find the size of the data types
  size_t FindByteSizeOfTensor(const torch::Tensor& tensor);
  size_t FindByteSizeOfString(const std::string& str);
  size_t FindByteSizeOfDictOfTensors(
      const std::map<std::string, torch::Tensor>& dict_of_tensors);

  void RecordANewData(size_t addr);

 public:
  SHMMessageQueue(size_t shm_size);
  ~SHMMessageQueue();

  // Main functions
  void WriteADictOfTensors(
      const std::map<std::string, torch::Tensor>& dict_of_tensors,
      size_t data_starting_addr);
  py::dict ReadADictOfTensors(size_t data_starting_addr);

  // Record the SHM id for attaching in different subprocesses
  int shmid;

  // For recording the operations of the message queue
  int GetNumOfSamples();
  std::atomic<size_t> case1;
  std::atomic<size_t> case2;
};

// A wrapper on top of SHMMessageQueue to allow the pickling/unpickling
// processing for mp.Process
class PicklableSHMMessageQueue {
 private:
  std::unique_ptr<SHMMessageQueue>
      shm_queue;  // It will be used by the creator of the SHM
  SHMMessageQueue*
      controller_in_shm_ptr;  // It will be used by the instances in mp.Process
  size_t data_starting_addr_per_process;  // It stores the starting address of
                                          // the attached shared memory (exclude
                                          // the controller/SHMMessageQueue) for
                                          // that particular process. The
                                          // absolute value will be varied for
                                          // different process as each process
                                          // has a new address space

 public:
  int shmid;
  PicklableSHMMessageQueue(size_t shmsize);
  PicklableSHMMessageQueue(
      int shmid,
      bool for_python_mp);  // Used in mp.Process during pickling and unpickling

  void WriteDictOfTensors(
      const std::map<std::string, torch::Tensor>& dict_of_tensors);
  py::dict ReadDictOfTensors();

  int GetNumOfSamples();

  // For monitoring the internal operations
  void Monitor();

  // For pybind
  int GetSHMid();
  size_t GetMPProcessSHMAddr();
};

}  // namespace pyg_distributed
#endif  // MESSAGEQUEUE_SHM_H
