#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
       for (size_t epoch = 0; epoch < m; epoch += batch){
        float *x = new float[batch * n];
        unsigned char *label = new unsigned char[batch];
        for (size_t i = 0; i < batch; i ++){
            for (size_t j = 0; j < n; j ++)
                x[i * n + j] = X[(epoch + i) * n + j];
            label[i] = y[epoch + i];
        }

        float *logits = new float[batch * k];
        for (size_t u = 0; u < batch; u ++)
            for (size_t v = 0; v < k; v ++){
                logits[u * k + v] = 0;
                for (size_t w = 0; w < n; w ++)
                    logits[u * k + v] += x[u * n + w] * theta[w * k + v];
            }
            
        for (size_t i = 0; i < batch; i ++){
            float sum = 0;
            for (size_t j = 0; j < k; j ++)
                sum += exp(logits[i * k + j]);
            for (size_t j = 0; j < k; j ++)
                logits[i * k + j] = exp(logits[i * k + j]) / sum;
        }

        int *I = new int[batch * k];
        for (size_t i = 0; i < batch * k; i ++) I[i] = 0;
        for (size_t i = 0; i < batch; i ++) I[i * k + label[i]] = 1;

        float *grad_theta = new float[n * k];
        for (size_t i = 0; i < n; i ++)
            for (size_t j = 0; j < k; j ++){
                grad_theta[i * k + j] = 0;
                for (size_t l = 0; l < batch; l ++)
                    grad_theta[i * k + j] += x[l * n + i] * (logits[l * k + j] - I[l * k + j]);
                theta[i * k + j] -= lr * grad_theta[i * k + j] / batch;           
            }

        delete[] x;
        delete[] label;
        delete[] logits;
        delete[] I;
        delete[] grad_theta;
        }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
