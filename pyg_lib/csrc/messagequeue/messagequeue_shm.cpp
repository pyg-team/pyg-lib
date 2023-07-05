// Implementing a thread-safe storage for multiple samplers and dataloaders

#include "./messagequeue_shm.h"

namespace pyg_distributed {
// Constructor -- will only be used in the main process
SHMMessageQueue::SHMMessageQueue(std::size_t shm_size) {
  // initialize a shared memory
  int shm_handle = shmget(IPC_PRIVATE, shm_size, 0666 | IPC_CREAT | IPC_EXCL);
  if (shm_handle == -1) {
    std::cout << "Error: shmget in shm_messagequeue.cpp";
  }
  shmid = shm_handle;  // record the SHM id for retrieval
  void* shmat_ptr = shmat(shm_handle, NULL, 0);
  if (*(int*)shmat_ptr == -1) {
    std::cout << "Error: shmat in shm_messagequeue.cpp";
  }

  // update the private variables
  reader_addr_relative.store(0);
  writer_addr_relative.store(0);
  shm_end_addr_relative = shm_size - sizeof(SHMMessageQueue) - 1;  // 0 based
  address_book_write_idx.store(0);
  address_book_read_idx.store(0);
  num_of_samples.store(0);
  size_of_shm = shm_size;

  // init the sempahores
  sem_init(&write_sem, 1, 1);
  sem_init(&address_book_sem, 1, 1);
  sem_init(&read_sem, 1, 1);

  // init for monitoring
  case1.store(0);
  case2.store(0);
}

SHMMessageQueue::~SHMMessageQueue() {
  sem_destroy(&write_sem);
  sem_destroy(&address_book_sem);
  sem_destroy(&read_sem);
  shmctl(shmid, IPC_RMID, NULL);
}

// put the starting address of a new data into an array
// reader will get from the array to start reading a new data
inline void SHMMessageQueue::RecordANewData(std::size_t addr) {
  sem_wait(&address_book_sem);
  address_book_arr[address_book_write_idx.load()] = addr;
  if (address_book_write_idx.load() + 1 >= MAX_SAMPLE_NUM) {
    address_book_write_idx.store(0);
  } else {
    address_book_write_idx.store(address_book_write_idx.load() + 1);
  }
  sem_post(&address_book_sem);
}

// a helper function to find the size of the serialized/1D tensor
inline size_t SHMMessageQueue::FindByteSizeOfTensor(
    const torch::Tensor& tensor) {
  std::size_t total_size = 0;
  total_size += sizeof(std::size_t);   // total length field
  total_size += sizeof(std::uint8_t);  // indicator field
  total_size += sizeof(std::size_t);   // shape num field
  total_size +=
      tensor.sizes().size() * sizeof(std::int64_t);  // shape values field
  total_size += sizeof(torch::ScalarType);           // data type field
  total_size += sizeof(std::size_t);                 // data length field
  total_size += tensor.nbytes();                     // payload size
  return total_size;
}

// a helper function to find the byte size of a string
inline size_t SHMMessageQueue::FindByteSizeOfString(const std::string& str) {
  return str.size() + sizeof(size_t);  // the size of the string and the  data
                                       // size field of the string
}

// a helper function to find the byte size of a dictionary of tensors (the
// number of key-pairs can be varied)
inline size_t SHMMessageQueue::FindByteSizeOfDictOfTensors(
    const std::map<std::string, torch::Tensor>& dict_of_tensors) {
  size_t total_size = 0;
  total_size += sizeof(size_t);  // the data size field of this block of data
  total_size +=
      sizeof(size_t);  // the field recording the number of key-value pairs

  // Loop through all the data
  for (const auto& item : dict_of_tensors) {
    total_size += FindByteSizeOfString(item.first);
    total_size += FindByteSizeOfTensor(item.second);
  }
  return total_size;
}

char* SHMMessageQueue::SerializeAString(void* shm_ptr, const std::string& str) {
  // init
  char* writer_ptr = static_cast<char*>(shm_ptr);
  writer_ptr += sizeof(size_t);  // Shift the size field

  for (const char& c : str) {
    *writer_ptr = c;
    writer_ptr++;
  }

  // write the size of this data
  *reinterpret_cast<size_t*>(shm_ptr) =
      reinterpret_cast<char*>(writer_ptr) - reinterpret_cast<char*>(shm_ptr);
  return writer_ptr;
}

char* SHMMessageQueue::SerializeATensor(void* shm_ptr,
                                        const torch::Tensor& tensor) {
  // init
  char* writer_ptr = static_cast<char*>(shm_ptr);

  // reserve a size_t place for recording the size of this data;
  for (std::size_t i = 0; i < sizeof(std::size_t) / sizeof(char); i++) {
    writer_ptr++;
  }

  // indicator
  *reinterpret_cast<std::uint8_t*>(writer_ptr) = 1;
  writer_ptr++;

  // shape number and values
  torch::IntArrayRef tensor_shape = tensor.sizes();
  std::size_t total_bytes_for_shape =
      tensor_shape.size() *
      sizeof(std::int64_t);  // c10::IntArrayRef is ArrayRef<int64_t>
  *reinterpret_cast<std::size_t*>(writer_ptr) = tensor_shape.size();
  writer_ptr += sizeof(std::size_t);
  memcpy(writer_ptr, tensor_shape.data(), total_bytes_for_shape);
  writer_ptr += total_bytes_for_shape;

  // data type and length
  *reinterpret_cast<torch::ScalarType*>(writer_ptr) = tensor.scalar_type();
  writer_ptr = writer_ptr + sizeof(torch::ScalarType);
  std::size_t payload_size = tensor.nbytes();
  *reinterpret_cast<std::size_t*>(writer_ptr) = payload_size;
  writer_ptr += sizeof(std::size_t);

  // raw tensor data
  memcpy(writer_ptr, tensor.data_ptr(), payload_size);
  writer_ptr += payload_size;

  // write the size of this data
  *reinterpret_cast<std::size_t*>(shm_ptr) =
      reinterpret_cast<std::size_t>(writer_ptr) -
      reinterpret_cast<std::size_t>(shm_ptr);

  return writer_ptr;
}

char* SHMMessageQueue::SerializeADictOfTensors(
    void* shm_ptr,
    const std::map<std::string, torch::Tensor>& dict_of_tensors) {
  // init
  char* writer_ptr = static_cast<char*>(shm_ptr);
  writer_ptr += sizeof(size_t);  // skip the data size field

  // write the number of key-values pairs of this dictionary
  *reinterpret_cast<size_t*>(writer_ptr) = dict_of_tensors.size();
  writer_ptr += sizeof(size_t);

  // Serialize all the key/str and values/tensor
  for (const auto& item : dict_of_tensors) {
    writer_ptr =
        SerializeAString(reinterpret_cast<void*>(writer_ptr), item.first);
    writer_ptr =
        SerializeATensor(reinterpret_cast<void*>(writer_ptr), item.second);
  }

  // Write the total size of this block of data into the data size field
  *reinterpret_cast<size_t*>(shm_ptr) =
      reinterpret_cast<char*>(writer_ptr) - reinterpret_cast<char*>(shm_ptr);
  return writer_ptr;
}

// the main write function
void SHMMessageQueue::WriteADictOfTensors(
    const std::map<std::string, torch::Tensor>& dict_of_tensors,
    size_t data_starting_addr) {
  sem_wait(&write_sem);
  std::size_t updated_writer_addr = 0;
  std::size_t data_size = FindByteSizeOfDictOfTensors(dict_of_tensors);
  void* writer_ptr;

  // Check the data size
  try {
    if (data_size > size_of_shm)
      throw data_size;  // Throw an exception when a problem arise
  } catch (size_t data_size) {
    std::cout << "ERROR: The data size of the message (" << data_size << ") "
              << "is larger than the shared memory (" << size_of_shm << ").";
  }
  // Recommend a better shared memory size
  if (data_size * 4 > size_of_shm) {
    std::cout << "Warning: It is recommended to initialize a larger shared "
                 "memory size for the MessageQueue."
              << std::endl;
  }

  // Note: In here we only allow writing happens when the writer leads
  //       This will help to remove the deadlock issue. (There is a case that
  //       writer is waiting for reader and reader is waiting for writer)
  // i.e. writer leads and remaining space is enough to write the data
  if ((writer_addr_relative.load() >= reader_addr_relative.load()) &&
      (writer_addr_relative.load() + data_size <= shm_end_addr_relative)) {
    writer_ptr = reinterpret_cast<void*>(data_starting_addr +
                                         writer_addr_relative.load());
    SerializeADictOfTensors(writer_ptr, dict_of_tensors);
    RecordANewData(writer_addr_relative.load());
    writer_addr_relative.store(writer_addr_relative.load() + data_size);
    case1.store(case1.load() + 1);
  } else {
    // Not enough space in the remaining space. Need to go back to the beginning
    // of the buffer to write. Wait for the reader to consume all the data
    // first. This can prevent data corruption
    while (num_of_samples.load() != 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    reader_addr_relative.store(0);
    writer_addr_relative.store(0);
    writer_ptr = reinterpret_cast<void*>(data_starting_addr);
    SerializeADictOfTensors(writer_ptr, dict_of_tensors);
    RecordANewData(0);
    writer_addr_relative.store(data_size);
    case2.store(case2.load() + 1);
  }

  num_of_samples.store(num_of_samples.load() + 1);
  sem_post(&write_sem);
  return;
}

// the main read function
py::dict SHMMessageQueue::ReadADictOfTensors(size_t data_starting_addr) {
  sem_wait(&read_sem);
  py::dict result;

  // block until a new data is generated
  while (num_of_samples.load() == 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // read the address book
  sem_wait(&address_book_sem);
  std::size_t read_addr_offset = address_book_arr[address_book_read_idx.load()];
  if (address_book_read_idx.load() + 1 >=
      MAX_SAMPLE_NUM) {  // reset the reader index if that is larger than the
                         // size of the address book array
    address_book_read_idx.store(0);
  } else {
    address_book_read_idx.store(address_book_read_idx.load() + 1);
  }
  void* actual_readaddr_ptr =
      reinterpret_cast<void*>(data_starting_addr + read_addr_offset);
  sem_post(&address_book_sem);

  // start deserializing the string from the shared memory
  result = DeSerializeADictOfTensors(actual_readaddr_ptr);

  // update metadata
  std::uint8_t data_size = *reinterpret_cast<std::size_t*>(
      actual_readaddr_ptr);  // read the size of the data (which is specified at
                             // the beginning of the bytes [data_size][tensor
                             // data......])
  reader_addr_relative.store(read_addr_offset + data_size);
  num_of_samples.store(num_of_samples.load() - 1);
  sem_post(&read_sem);

  return std::move(result);
}

// deserialize the string
inline std::string SHMMessageQueue::DeserializeAString(void* shm_ptr) {
  char* reader_ptr = static_cast<char*>(shm_ptr);
  std::string result = "";

  // skip the size of the data
  size_t str_size = *reinterpret_cast<size_t*>(reader_ptr) -
                    sizeof(size_t);  // minus the total size field
  reader_ptr += sizeof(size_t);

  for (uint8_t loop = 0; loop < str_size; loop++) {
    result += *reader_ptr;
    reader_ptr++;
  }

  return result;
}

// deserialize the tensor
// the data_ptr should point at the "indicator" of the tensor
inline torch::Tensor SHMMessageQueue::DeserializeATensor(void* shm_ptr) {
  char* reader_ptr = static_cast<char*>(shm_ptr);
  torch::Tensor reconstructed_tensor;

  // skip the size of the data
  for (std::size_t i = 0; i < sizeof(std::size_t) / sizeof(char); i++) {
    reader_ptr++;
  }

  // read the indicator
  std::uint8_t indicator = *reinterpret_cast<std::uint8_t*>(reader_ptr);
  reader_ptr++;

  if (indicator == 1) {
    // shape number and values
    std::size_t torch_shape_array_size =
        *reinterpret_cast<std::size_t*>(reader_ptr);
    reader_ptr += sizeof(std::size_t);
    torch::IntArrayRef tensor_shapes{
        reinterpret_cast<std::int64_t*>(reader_ptr), torch_shape_array_size};
    reader_ptr += torch_shape_array_size * sizeof(std::int64_t);

    // data type and length
    torch::ScalarType data_type =
        *reinterpret_cast<torch::ScalarType*>(reader_ptr);
    reader_ptr += sizeof(torch::ScalarType);
    std::size_t payload_size = *reinterpret_cast<std::size_t*>(reader_ptr);
    reader_ptr += sizeof(std::size_t);

    // construct the tensor from the above metadata
    reconstructed_tensor = torch::from_blob(
        reader_ptr, tensor_shapes,
        torch::TensorOptions().dtype(data_type).device(
            torch::kCPU));  // creates a tensor view of the data
    reader_ptr += payload_size;
  } else {
    return torch::zeros({0});  // indicates that it does not contain data
  }

  return reconstructed_tensor.clone()
      .detach();  // return a new instance (as the shm buffer will be overwrote)
}

// deserialize a dictionary which contains single/multiple tensors
py::dict SHMMessageQueue::DeSerializeADictOfTensors(void* shm_ptr) {
  // init
  char* reader_ptr = static_cast<char*>(shm_ptr);
  py::dict result;

  // read the metadata
  size_t total_size = *reinterpret_cast<size_t*>(reader_ptr);
  reader_ptr += sizeof(size_t);
  size_t num_of_keyvalue_pair = *reinterpret_cast<size_t*>(reader_ptr);
  reader_ptr += sizeof(size_t);

  for (size_t i = 0; i < num_of_keyvalue_pair; i++) {
    std::string key = DeserializeAString(reinterpret_cast<void*>(reader_ptr));
    reader_ptr += *reinterpret_cast<size_t*>(reader_ptr);
    torch::Tensor value =
        DeserializeATensor(reinterpret_cast<void*>(reader_ptr));
    reader_ptr += *reinterpret_cast<size_t*>(reader_ptr);
    result[py::cast(std::move(key))] = std::move(value);
  }
  return std::move(result);
}

int SHMMessageQueue::GetNumOfSamples() {
  return num_of_samples.load();
}

// The main constructor that is used in main process
PicklableSHMMessageQueue::PicklableSHMMessageQueue(std::size_t shmsize) {
  try {
    if (shmsize < 134217728) {  // i.e. 128MB
      throw(shmsize);
    }
  } catch (std::size_t shmsize) {
    std::cout << "ERROR: The shared memory size should be larger than 128MB "
                 "(i.e. 134217728). It is now: "
              << shmsize << std::endl;
    return;
  }

  shm_queue = std::make_unique<SHMMessageQueue>(shmsize);
  shmid = shm_queue->shmid;

  // copy the controller to the shared memory
  void* shm_ptr = shmat(shmid, NULL, 0);
  void* shm_queue_ptr = reinterpret_cast<void*>(shm_queue.get());
  memcpy(shm_ptr, shm_queue_ptr, sizeof(SHMMessageQueue));

  controller_in_shm_ptr = reinterpret_cast<SHMMessageQueue*>(shm_ptr);
  data_starting_addr_per_process = reinterpret_cast<size_t>(shm_ptr);
  data_starting_addr_per_process +=
      sizeof(SHMMessageQueue);  // Avoid the serializer/deserializer corrupt the
                                // controller/SHMMessageQueue

  std::cout << "Initialized a SHM. SHM ID: " << shmid << std::endl;
}

// This constructor will be called when passing this object through
// mp.Process(.... args(this_object)). It will use the 'shmid' and 'attach_addr'
// to construct a pointer of SHMMessageQueue. The pointer will point at the
// shared memory. This is to ensure that multiprocesses are using the same
// controller/SHMMessagequeue to read/write tensors. We leverage the mechanism
// of py::pickle to achieve this and avoid the limitations posed by
// mp.Process().
PicklableSHMMessageQueue::PicklableSHMMessageQueue(int _shmid,
                                                   bool for_python_mp) {
  void* shm_ptr = shmat(_shmid, (void*)0, 0);
  if (*(int*)shm_ptr == -1) {
    std::cout << "Error: shmat in PicklableSHMMessageQueue";
  }
  shmid = _shmid;

  // Load the SHMMessageQueue from the shared memory
  controller_in_shm_ptr = reinterpret_cast<SHMMessageQueue*>(shm_ptr);
  data_starting_addr_per_process = reinterpret_cast<size_t>(shm_ptr);
  data_starting_addr_per_process +=
      sizeof(SHMMessageQueue);  // Avoid the serializer/deserializer corrupt the
                                // controller/SHMMessageQueue
}

int PicklableSHMMessageQueue::GetSHMid() {
  return shmid;
}

size_t PicklableSHMMessageQueue::GetMPProcessSHMAddr() {
  std::size_t addr = reinterpret_cast<std::size_t>(controller_in_shm_ptr);
  return addr;
}

int PicklableSHMMessageQueue::GetNumOfSamples() {
  return controller_in_shm_ptr->GetNumOfSamples();
}

void PicklableSHMMessageQueue::WriteDictOfTensors(
    const std::map<std::string, torch::Tensor>&
        dict_of_tensors) {  // Csat the Python dictionary to a C++ type for
                            // processing
  return controller_in_shm_ptr->WriteADictOfTensors(
      dict_of_tensors, data_starting_addr_per_process);
}

py::dict
PicklableSHMMessageQueue::ReadDictOfTensors() {  // Map the dictionary to a C++
                                                 // type for processing
  return controller_in_shm_ptr->ReadADictOfTensors(
      data_starting_addr_per_process);
}

// for monitoring the operation of the message queue
void PicklableSHMMessageQueue::Monitor() {
  std::cout << "case1 : " << controller_in_shm_ptr->case1.load()
            << " case2 : " << controller_in_shm_ptr->case2.load() << std::endl;
}

};  // namespace pyg_distributed
