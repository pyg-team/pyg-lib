#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <torch/extension.h>
#include "./messagequeue_shm.h"
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "MessageQueue for PyG distributed trainings";

  py::class_<pyg_distributed::PicklableSHMMessageQueue>(m, "shm")
    .def(py::init<size_t>(), py::arg("shm_size"))
    .def("GetSHMid", &pyg_distributed::PicklableSHMMessageQueue::GetSHMid)
    .def("GetMPProcessSHMAddr", &pyg_distributed::PicklableSHMMessageQueue::GetMPProcessSHMAddr)
    .def("GetNumOfSamples", &pyg_distributed::PicklableSHMMessageQueue::GetNumOfSamples)
    .def("WriteDictOfTensors", &pyg_distributed::PicklableSHMMessageQueue::WriteDictOfTensors)
    .def("ReadDictOfTensors", &pyg_distributed::PicklableSHMMessageQueue::ReadDictOfTensors)
    .def("Monitor", &pyg_distributed::PicklableSHMMessageQueue::Monitor)
    .def(py::pickle(
      [](const pyg_distributed::PicklableSHMMessageQueue& q) { 
          return py::make_tuple(q.shmid, true);
      },
      [](py::tuple t) {                                       
          return pyg_distributed::PicklableSHMMessageQueue{t[0].cast<int>(), t[1].cast<bool>()};
      }
    ));
}
