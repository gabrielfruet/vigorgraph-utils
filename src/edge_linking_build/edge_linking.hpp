#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>


std::vector<cv::Point> find_neighbors(int i, int j, const cv::Size& size);
CV_EXPORTS_W std::vector<std::vector<cv::Point>> edge_linking(py::array_t<uint8_t>& img);

