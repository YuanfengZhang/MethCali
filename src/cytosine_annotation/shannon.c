#include <Python.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

// 1. Shannon Entropy
static PyObject* shannon_entropy(PyObject* self, PyObject* args) {
    const char* seq;
    if (!PyArg_ParseTuple(args, "s", &seq)) {
        return NULL;
    }

    int len_seq = strlen(seq);
    if (len_seq == 0) {
        return Py_BuildValue("d", 0.0);
    }

    int base_count[4] = {0};
    for (int i = 0; i < len_seq; i++) {
        char base = seq[i];
        if (base == 'A') base_count[0]++;
        else if (base == 'C') base_count[1]++;
        else if (base == 'G') base_count[2]++;
        else if (base == 'T') base_count[3]++;
    }

    double entropy = 0.0;
    for (int i = 0; i < 4; i++) {
        if (base_count[i] > 0) {
            double freq = (double)base_count[i] / len_seq;
            entropy -= freq * log2(freq);
        }
    }

    return Py_BuildValue("d", entropy);
}

// Define the methods in the module
static PyMethodDef SequenceMethods[] = {
    {"shannon_entropy", shannon_entropy, METH_VARARGS, "Calculate Shannon entropy of a DNA sequence"},
    {NULL, NULL, 0, NULL}
};

// Define the module
static struct PyModuleDef seq_comp = {
    PyModuleDef_HEAD_INIT,
    "shannon",
    NULL,
    -1,
    SequenceMethods
};

// Initialize the module
PyMODINIT_FUNC PyInit_shannon(void) {
    return PyModule_Create(&seq_comp);
}

// To compile:
// gcc -O3 -Wall -shared -std=c99
//   -fPIC $(python3 -m pybind11 --includes)
//   -I/home/zhangyuanfeng/mambaforge/pkgs/python-3.12.2-hab00c5b_0_cpython/include/python3.12
//   shannon.c -o shannon.so
