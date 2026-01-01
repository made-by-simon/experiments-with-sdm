#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "kanerva_sdm.h"

namespace py = pybind11;

PYBIND11_MODULE(sdm_cpp, m) {
    m.doc() = "C++ implementation of Kanerva's Sparse Distributed Memory";

    py::class_<KanervaSDM>(m, "KanervaSDM")
        .def(py::init<int, int, int, int, unsigned int>(),
             py::arg("address_dimension"),
             py::arg("memory_dimension"),
             py::arg("num_locations"),
             py::arg("activation_threshold"),
             py::arg("random_seed") = 42,
             "Initialize Kanerva SDM")
        
        .def("write", &KanervaSDM::write,
             py::arg("address"),
             py::arg("memory"),
             "Write a memory to an address")
        
        .def("read", &KanervaSDM::read,
             py::arg("address"),
             "Read a memory from an address")
        
        .def("erase_memory", &KanervaSDM::erase_memory,
             "Erase memory matrix but preserve hard locations")
        
        .def("get_address_dimension", &KanervaSDM::get_address_dimension,
             "Get address dimension (N)")
        
        .def("get_memory_dimension", &KanervaSDM::get_memory_dimension,
             "Get memory dimension (U)")
        
        .def("get_num_locations", &KanervaSDM::get_num_locations,
             "Get number of hard locations (M)")
        
        .def("get_activation_threshold", &KanervaSDM::get_activation_threshold,
             "Get activation threshold (H)")
        
        .def("get_memory_count", &KanervaSDM::get_memory_count,
             "Get number of stored memories (T)");
}