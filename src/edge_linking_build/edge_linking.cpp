#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/typing.h>
#include "pybind11/numpy.h"
#include <vector>
#include <array>

namespace py = pybind11;

std::vector<cv::Point> find_neighbors(int i, int j, const cv::Size& size) {
    cv::Point pos(j,i);
    std::array<cv::Point, 8> directions = {{
        {1, -1}, {1, 0}, {1, 1}, {0, -1}, {0, 1}, {-1, -1}, {-1, 0}, {-1, 1}
    }};
    std::vector<cv::Point> valid_neighbors;
    
    for (const auto& dir : directions) {
        cv::Point newpos = dir + pos;
        if (newpos.x >= 0 && newpos.x < size.width && newpos.y >= 0 && newpos.y < size.height) {
            valid_neighbors.push_back(newpos);
        }
    }
    return valid_neighbors;
}

std::vector<py::array_t<uint32_t>> edge_linking(py::array_t<uint8_t>& input) {
    py::array_t<uint8_t> input_clone = input.attr("copy")().cast<py::array_t<uint8_t>>();
    cv::Mat A(input.shape(0), input.shape(1), CV_8UC1, (uchar*)input_clone.data()); // Copy the input array
    std::vector<std::vector<cv::Point>> links;
    cv::Size size = A.size();
    
    for (int i = 0; i < size.height; ++i) {
        for (int j = 0; j < size.width; ++j) {
            if (A.at<uchar>(i,j) == 255) {
                A.at<uchar>(i,j) = 0;
                int k = i, l = j;
                std::vector<cv::Point> new_link({cv::Point(j,i)});
                bool available_paths = true;
                
                while (available_paths) {
                    auto neighbors = find_neighbors(k, l, size);
                    available_paths = false;
                    for (const auto neighbor : neighbors) {
                        int p = neighbor.y, q = neighbor.x;
                        if (A.at<uchar>(p,q) == 255) {
                            A.at<uchar>(p,q) = 0;
                            new_link.push_back(cv::Point(q,p));
                            k = p;
                            l = q;
                            available_paths = true;
                            break;
                        }
                    }
                }
                links.push_back(new_link);
            }
        }
    }

    std::vector<py::array_t<uint32_t>> links_converted;
    for (const auto& link : links) {
        std::vector<uint32_t> flat_link;
        for (const auto& pt : link) {
            flat_link.push_back(pt.x);
            flat_link.push_back(pt.y);
        }
        py::array_t<uint32_t> array_link({int(link.size()), 2}, flat_link.data());
        links_converted.push_back(array_link);
    }
    return links_converted;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("edge_linking", &edge_linking, "A function that adds two numbers");
}
