#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// #include "net_model.h"

using namespace std;


typedef enum net_status {
    NET_STATUS_ERROR = 0,
    NET_STATUS_OK,
} NET_STATUS;

typedef enum layer_type {
    LAYER_TYPE_ERROR = 0,
    LAYER_TYPE_INPUT,
    LAYER_TYPE_CONV,
    LAYER_TYPE_MAXPOOL,
    LAYER_TYPE_ROUTE,
    LAYER_TYPE_REORG,
    LAYER_TYPE_YOLO
} LAYER_TYPE;

typedef enum batch_type {
    ACTIVATION_TYPE_ERROR = 0,
    ACTIVATION_TYPE_LEAKY,
    ACTIVATION_TYPE_LINEAR,
} ACTIVATION_TYPE;

// typedef struct LAYER_INFO layer_info;

typedef struct layer_info {
    LAYER_TYPE type;
    int8_t layers[3];
    uint16_t nb_of_filters;
    uint8_t kernel_size;
    uint8_t stride;
    uint8_t pad;
    uint8_t batch_normalize;
    ACTIVATION_TYPE activation;
    float bn_weights[8192];
    float conv_weights[8192];
    float conv_bias[8192];
    std::vector < cv::Mat > dst;
    // const LAYER_INFO * next;
} LAYER_INFO;

typedef struct net_info {
    cv::Mat src;
    cv::Mat dst;
} NET_INFO;



typedef struct layer_type_mapping {
    string str;
    LAYER_TYPE type;
} LAYER_TYPE_MAPPING;

uint8_t _nb_of_layer = 32;
LAYER_INFO _layer_infos[32];

int8_t conv_layer_counter = 0;
int8_t norm_layer_counter = 0;
int8_t leaky_re_lu_layer_counter = -1;
int8_t max_pooling2d_layer_counter = -1;


NET_STATUS NET_CreateModel(void) {
    NET_STATUS _status = NET_STATUS_ERROR;
    //---------------------------------
    int i;
    string _line;
    string _str;
    size_t _pos;
    ifstream _myfile;
    char _bracket = 0x5B;
    string _new_line = "\n";
    string _comment = "#";
    int _counter = 1;
    // uint8_t _nb_of_layer = 32;
    // LAYER_INFO _layer_infos[32];
    LAYER_TYPE_MAPPING _layer_type_mapping[] = {
        {"convolutional", LAYER_TYPE_CONV   },
        {"maxpool"      , LAYER_TYPE_MAXPOOL},
        {"route"        , LAYER_TYPE_ROUTE  },
        {"reorg"        , LAYER_TYPE_REORG  },
        {"region"       , LAYER_TYPE_YOLO   },
    };
    //---
    _layer_infos[0].type = LAYER_TYPE_INPUT;
    for (i = 0; i < 3; i++) {
        _layer_infos[_counter].layers[i] = 0;
    }
    _layer_infos[0].nb_of_filters = 3;
    _layer_infos[0].stride = 0;
    _layer_infos[0].pad = 0;
    _layer_infos[0].batch_normalize = 0;
    _layer_infos[0].activation = ACTIVATION_TYPE_ERROR;
    //---
    _myfile.open("yolov2.cfg");
    // _myfile.open("yolov3.cfg");
    
    if (_myfile.is_open()) {
        //---
        while (getline(_myfile, _line) && _counter < _nb_of_layer) {
            cout << "\n" << _line;
            if (_line.find("net") == 1 ||
                _line.find(_new_line) == 0 ||
                _line.find(_comment) == 0)
            {
                cout << "\n error";
            }
            else if (_line[0] == _bracket) {
                //---
                // cout << "\n" << "[";
                _pos = _line.find("]");
                _pos--;
                // cout << "\n" << _pos;
                cout << "\n" << _line.substr(1, _pos);
                //---
                _layer_infos[_counter].type = LAYER_TYPE_ERROR;
                for (i = 0; i < 3; i++) {
                    _layer_infos[_counter].layers[i] = 0;
                }
                _layer_infos[_counter].nb_of_filters = 0;
                _layer_infos[_counter].stride = 0;
                _layer_infos[_counter].pad = 0;
                _layer_infos[_counter].batch_normalize = 0;
                _layer_infos[_counter].activation = ACTIVATION_TYPE_ERROR;
                memset(_layer_infos[_counter].bn_weights, 0, 8192);
                memset(_layer_infos[_counter].conv_bias, 0, 8192);
                memset(_layer_infos[_counter].conv_weights, 0, 8192);
                //---
                for (i = 0; i < sizeof(_layer_type_mapping)/sizeof(_layer_type_mapping[0]); i++) {
                    if (_line.find(_layer_type_mapping[i].str) == 1) {
                        _layer_infos[_counter].type = _layer_type_mapping[i].type;
                        break;
                    }
                }
                //---
                while (getline(_myfile, _line)) {
                    cout << "\n" << _line;

                    if (_line.find("batch_normalize") == 0) {
                        _layer_infos[_counter].batch_normalize = 1;
                    }
                    else if (_line.find("filters") == 0) {
                        _pos = _line.find("=");
                        _pos++;
                        _str = _line.substr(_pos);
                        _layer_infos[_counter].nb_of_filters = stoi(_str);
                    }
                    else if (_line.find("size") == 0) {
                        _pos = _line.find("=");
                        _pos++;
                        _str = _line.substr(_pos);
                        _layer_infos[_counter].kernel_size = stoi(_str);
                    }
                    else if (_line.find("stride") == 0) {
                        _pos = _line.find("=");
                        _pos++;
                        _str = _line.substr(_pos);
                        _layer_infos[_counter].stride = stoi(_str);
                    }
                    else if (_line.find("pad") == 0) {
                        _pos = _line.find("=");
                        _pos++;
                        _str = _line.substr(_pos);
                        _layer_infos[_counter].pad = stoi(_str);
                    }
                    else if (_line.find("activation") == 0) {
                        _pos = _line.find("=");
                        _pos++;
                        _str = _line.substr(_pos);
                        if (_str.compare("leaky") == 0) {
                            _layer_infos[_counter].activation = ACTIVATION_TYPE_LEAKY;
                        }
                        else if (_str.compare("linear") == 0) {
                            _layer_infos[_counter].activation = ACTIVATION_TYPE_LINEAR;
                        }
                    }
                    else if (_line.find("layers") == 0) {
                        _layer_infos[_counter].layers[0] = 0;
                        _layer_infos[_counter].layers[1] = 0;
                        _layer_infos[_counter].layers[2] = 0;

                        size_t  _len = _line.size();
                        _pos = _line.find("=");
                        size_t _pos_2 = _line.find(",");
                        if (_pos_2 > _len) {
                            _pos_2 = _len;
                        }
                        
                        string _mystr = _line.substr(_pos+1, 2);
                        cout << "\n" << _mystr;

                        _layer_infos[_counter].layers[0] = stoi(_mystr);

                        if (_pos_2 < _len) {
                            _mystr = _line.substr(_pos_2+1, 2);
                            cout << "\n" << _mystr;
                            _layer_infos[_counter].layers[1] = stoi(_mystr);
                        }
                    }
                    else {
                        break;
                    }
                }
                //---
                _counter++;
                //---
            }
        }
        //---
        _myfile.close();
        //---
        _status = NET_STATUS_OK;
        //---
    } else {
        cout << "Unable to open file!";
    }
    //---------------------------------
    return _status;
}

NET_STATUS NET_LoadWeights(void) {
    NET_STATUS _status = NET_STATUS_ERROR;
    //---------------------------------
    ifstream _myfile;
    int32_t _read_buffer[5] = {0};
    const char * _yolo_cfg_file_path = "./yolov2.cfg";
    const char * _yolo_weights_file_path = "./yolov2.weights";
    // const char * _yolo_cfg_file_path = "./yolov3.cfg";
    // const char * _yolo_weights_file_path = "./yolov3.weights";
    int16_t _yolo_cfg_file_header_size = 4; // 5;
    int16_t _nb_of_layers = 0;
    int16_t _nb_of_filters = 0;
    float * _bn_weights = 0;
    float * _conv_weights = 0;
    float * _conv_bias = 0;
    ofstream _myfile_2;
    const char * _norm_weights_file_path = "./outputs/weights_norm.bin";
    const char * _conv_weights_file_path = "./outputs/weights_conv.bin";
    //---
    _myfile.open(_yolo_weights_file_path, ios::binary);
    if (_myfile.is_open()) {
        //---
        cout << "File is opened!" << "\n";
        //---
        //read header
        //---
        for (int i = 0; i < _yolo_cfg_file_header_size; i++) {
            _myfile.read(reinterpret_cast<char *>(&_read_buffer[i]), sizeof(int32_t));
        }
        //---
        _nb_of_layers = _nb_of_layer;
        for (int i = 1; i < _nb_of_layers; i++) {
            if (_layer_infos[i].type == LAYER_TYPE_CONV) {
                //---
                if (_layer_infos[i].batch_normalize == 1) {
                    //--- [gamma, beta, mean, variance] ---
                    _nb_of_filters = 4 * _layer_infos[i].nb_of_filters;
                    // _bn_weights = (float *)malloc(_nb_of_filters);
                    _bn_weights = _layer_infos[i].bn_weights;
                    for (int j = 0; j < _nb_of_filters; j++) {
                        _myfile.read(reinterpret_cast<char *>(_bn_weights + j), sizeof(float));
                    }
                    //---
                    _myfile_2.open(_norm_weights_file_path, ios::out | ios::binary);
                    if (_myfile_2.is_open()) {
                        _myfile_2.write(reinterpret_cast<char *>(_bn_weights), _nb_of_filters * sizeof(float));
                        _myfile_2.close();
                    }
                    else {
                        cout << "Unable to open file!";
                    }
                    //---
                }
                else {
                    _nb_of_filters = _layer_infos[i].nb_of_filters;
                    // _conv_bias = (float *)malloc(_nb_of_filters);
                    _conv_bias = _layer_infos[i].conv_bias;
                    for (int j = 0; j < _nb_of_filters; j++) {
                        _myfile.read(reinterpret_cast<char *>(_conv_bias + j), sizeof(float));
                    }
                }
                //---
                _nb_of_filters = 3 * _layer_infos[i].nb_of_filters * _layer_infos[i].kernel_size * _layer_infos[i].kernel_size;
                // _conv_weights = (float *)malloc(_nb_of_filters);
                _conv_weights = _layer_infos[i].conv_weights;
                for (int j = 0; j < _nb_of_filters; j++) {
                    _myfile.read(reinterpret_cast<char *>(_conv_weights + j), sizeof(float));
                }
                //---
                _myfile_2.open(_conv_weights_file_path, ios::out | ios::binary);
                if (_myfile_2.is_open()) {
                    _myfile_2.write(reinterpret_cast<char *>(_conv_weights), _nb_of_filters * sizeof(float));
                    _myfile_2.close();
                }
                else {
                    cout << "Unable to open file!";
                }
                //---
            }
        }
        //---
        _myfile.close();
        //---
    } else {
        cout << "Unable to open file!";
    }
    //---------------------------------
    return _status;
}

/*NET_STATUS NET_Padding(void) {
    NET_STATUS _status = NET_STATUS_ERROR;
    //---------------------------------
    cv::Mat borderd_img;
    cv::Scalar value = 0;
    cv::Mat in_img;
    //---
    in_img = cv::imread("./image/lenna.png", cv::IMREAD_GRAYSCALE);
    cv::copyMakeBorder(in_img, borderd_img, 3, 3, 3, 3, cv::BORDER_CONSTANT, value);
    cv::imwrite("./image/output_4.png", borderd_img);
    //---------------------------------
    return _status;
}*/
NET_STATUS NET_Padding(int _layer_num, NET_INFO * _net_info, LAYER_INFO * _layer_info) {
    NET_STATUS _status = NET_STATUS_ERROR;
    //---------------------------------
    int d;
    int _prev_layer_num = _layer_num - 1;
    int _in_deep = _layer_infos[_prev_layer_num].nb_of_filters;
    int _pad = _layer_info->pad;
    cv::Scalar _value = 0;
    //---
    if (_layer_infos[_prev_layer_num].type == LAYER_TYPE_MAXPOOL){
        _in_deep = _layer_infos[_prev_layer_num - 1].nb_of_filters;
    }
    //---
    for (d = 0; d < _in_deep; d++) {
        cv::copyMakeBorder(_layer_infos[_layer_num - 1].dst[d], _layer_infos[_layer_num - 1].dst[d], _pad, _pad, _pad, _pad, cv::BORDER_CONSTANT, _value);
    }
    //---
    _status = NET_STATUS_OK;
    //---------------------------------
    return _status;
}

/*NET_STATUS NET_Conv(void) {
    NET_STATUS _status = NET_STATUS_ERROR;
    //---------------------------------
    int i, j, k, l;
    uchar sum;
    int stride = 1;
    cv::Mat in_img;
    //---

    // float kernel_data[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    uchar kernel_data[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
    cv::Mat kernel = cv::Mat(3, 3, CV_8U, kernel_data);
    std::cout << "kernel = " << std::endl << kernel << std::endl;

    in_img = cv::imread("./image/lenna.png", cv::IMREAD_GRAYSCALE);
    uchar me = 0;
    uchar me_1 = 0;
    uchar me_2 = 0;
    
    cv::Mat out_img{in_img.size(), in_img.type()};
    
    for (i = 0; i < in_img.size().width; i+=stride) {
        for (j = 0; j < in_img.size().height; j+=stride) {
            sum = 0;
            for (k = -1; k <= 1; k++) {
                for (l = -1; l <= 1; l++) {
                    if (((i + k) >= 0 && (i + k) < in_img.size().width) &&
                        ((j + l) >= 0 && (j + l) < in_img.size().width))
                    {
                        me_1 = in_img.at<uchar>(i + k, j + l);
                        me_2 = kernel.at<uchar>(k + 1, l + 1);
                        sum += me_1 * me_2;
                    }
                }
            }
            out_img.at<uchar>(i, j) = (uchar)sum;
        }
    }

    cv::imwrite("./image/output_2.png", out_img);
    //---------------------------------
    return _status;
}*/
NET_STATUS NET_Conv(int _layer_num, NET_INFO * _net_info, LAYER_INFO * _layer_info) {
    NET_STATUS _status = NET_STATUS_ERROR;
    //---------------------------------
    int i, j, k, l, m, n, d;
    std::vector< float > _sum;
    float _me_1 = 0;
    float _me_2 = 0;
    int _stride = _layer_info->stride;
    int _kernel_size = _layer_info->kernel_size;
    int _pad = _layer_info->pad;
    // float _test_kernel_data[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
    std::vector < std::vector< float > > _kernel_data;
    std::vector < cv::Mat > _kernel;
    cv::Size _matsize;
    int _prev_layer_num = _layer_num - 1;
    uint16_t _deep = _layer_infos[_prev_layer_num].nb_of_filters;
    ofstream _myfile;
    char _file_path[200] = {0};
    char _str_m[10] = {0};
    //---
    if (_layer_infos[_prev_layer_num].type == LAYER_TYPE_MAXPOOL){
        _deep = _layer_infos[_prev_layer_num - 1].nb_of_filters;
    }
    //---
    _sum.resize(_deep);
    _kernel.resize(_deep);
    _kernel_data.resize(_deep);
    for (i = 0; i < _deep; i++) {
        _kernel_data[i].resize(_layer_info->kernel_size * _layer_info->kernel_size);
    }
    //---
    _matsize.width = _layer_infos[_prev_layer_num].dst[0].size().width - (2 * _pad);
    _matsize.height = _layer_infos[_prev_layer_num].dst[0].size().height - (2 * _pad);
    for (m = 0; m < _layer_info->nb_of_filters; m++) {
        _layer_info->dst[m] = cv::Mat{_matsize, CV_32F}; // _net_info->src.type()
    }
    //---
    for (m = 0; m < _layer_info->nb_of_filters; m++) {
        //---
        for (d = 0; d < _deep; d++) {
            for (n = 0; n < (_layer_info->kernel_size * _layer_info->kernel_size); n++) {
                _kernel_data[d][n] = _layer_info->conv_weights[m * _deep * (_layer_info->kernel_size * _layer_info->kernel_size) + (d * _layer_info->kernel_size * _layer_info->kernel_size) + n];
            }
        }
        //---
        for (d = 0; d < _deep; d++) {
            _kernel[d] = cv::Mat(_layer_info->kernel_size, _layer_info->kernel_size, CV_32F, _kernel_data[d].data());
            // std::cout << "kernel = " << std::endl << _kernel[d] << std::endl;
        }
        //---
        // _layer_info->dst[m] = cv::Mat{_net_info->src.size(), _net_info->src.type()};
        //---
        for (i = _pad; i <= _layer_infos[_prev_layer_num].dst[0].size().width - (2 * _pad); i += _stride) {
            for (j = _pad; j <= _layer_infos[_prev_layer_num].dst[0].size().height - (2 * _pad); j += _stride) {
                //---
                for (d = 0; d < _deep; d++) {
                    _sum[d] = 0;
                }
                //---
                for (k = -1; k <= 1; k++) {
                    for (l = -1; l <= 1; l++) {
                        if (((i + k) >= 0 && (i + k) < _layer_infos[_prev_layer_num].dst[0].size().width) &&
                            ((j + l) >= 0 && (j + l) < _layer_infos[_prev_layer_num].dst[0].size().height))
                        {
                            for (d = 0; d < _deep; d++) {
                                _me_1 = (float)(_layer_infos[_prev_layer_num].dst[d].at<float>(i + k, j + l));
                                _me_2 = _kernel[d].at<float>(k + 1, l + 1);
                                // _me_2 = _kernel[d].at<float>(_kernel_size - 1 - (k + 1), _kernel_size - 1 - (l + 1));
                                _sum[d] += _me_1 * _me_2;
                            }
                        }
                    }
                }
                //---
                _layer_info->dst[m].at<float>(i - _pad, j - _pad) = 0;
                for (d = 0; d < _deep; d++) {
                    _layer_info->dst[m].at<float>(i - _pad, j - _pad) += (float)_sum[d];
                }
                //---
            }
        }
        //---
    }
    //---
    for (m = 0; m < _layer_info->nb_of_filters; m++) {
        sprintf(_file_path, "./outputs/conv_");
        sprintf(_str_m, "%d", conv_layer_counter);
        strcat(_file_path, _str_m);
        strcat(_file_path, "_d_");
        sprintf(_str_m, "%d", m);
        strcat(_file_path, _str_m);
        strcat(_file_path, ".bin");
        _myfile.open(_file_path, ios::out | ios::binary);
        if (_myfile.is_open()) {
            _myfile.write(reinterpret_cast<char *>(_layer_info->dst[m].data), _layer_info->dst[m].size().width * _layer_info->dst[m].size().height * sizeof(float));
            _myfile.close();
        }
        else {
            cout << "Unable to open file!";
        }
    }
    //---
    _status = NET_STATUS_OK;
    //---------------------------------
    return _status;
}

NET_STATUS NET_BatchNorm(int _layer_num, NET_INFO * _net_info, LAYER_INFO * _layer_info) {
    NET_STATUS _status = NET_STATUS_ERROR;
    //---------------------------------
    float _gamma = 1;
    float _beta = 0;
    float _mean = 0;
    float _var = 1;
    float _epsilon = 0.001; // 0.00001
    ofstream _myfile;
    char _file_path[200] = {0};
    char _str_m[10] = {0};
    //---
    for (int m = 0; m < _layer_info->nb_of_filters; m++) {
        //---
        _gamma = (float)_layer_info->bn_weights[_layer_info->nb_of_filters * 1 + m];
        _beta  = (float)_layer_info->bn_weights[_layer_info->nb_of_filters * 0 + m];
        _mean  = (float)_layer_info->bn_weights[_layer_info->nb_of_filters * 2 + m];
        _var   = (float)_layer_info->bn_weights[_layer_info->nb_of_filters * 3 + m];
        float _me = 0;
        //---
        for (int i = 0; i < _layer_info->dst[m].size().width; i++) {
            for (int j = 0; j < _layer_info->dst[m].size().height; j++) {
                _me = _layer_info->dst[m].at<float>(i, j);
                _layer_info->dst[m].at<float>(i, j) = _gamma * (_layer_info->dst[m].at<float>(i, j) - _mean) / sqrt(_var + _epsilon) + _beta;
                _me = _layer_info->dst[m].at<float>(i, j);
            }
        }
        //---
    }
    //---
    for (int m = 0; m < _layer_infos[_layer_num].nb_of_filters; m++) {
        sprintf(_file_path, "./outputs/norm_");
        sprintf(_str_m, "%d", norm_layer_counter);
        strcat(_file_path, _str_m);
        strcat(_file_path, "_d_");
        sprintf(_str_m, "%d", m);
        strcat(_file_path, _str_m);
        strcat(_file_path, ".bin");
        _myfile.open(_file_path, ios::out | ios::binary);
        if (_myfile.is_open()) {
            _myfile.write(reinterpret_cast<char *>(_layer_infos[_layer_num].dst[m].data), _layer_infos[_layer_num].dst[m].size().width * _layer_infos[_layer_num].dst[m].size().height * sizeof(float));
            _myfile.close();
        }
        else {
            cout << "Unable to open file!";
        }
    }
    //---
    _status = NET_STATUS_OK;
    //---------------------------------
    return _status;
}

/*NET_STATUS NET_Activation(void) {
    NET_STATUS _status = NET_STATUS_ERROR;
    //---------------------------------
    cv::Mat in_img;
    // int a = 1;
    //---
    in_img = cv::imread("./image/lenna.png", cv::IMREAD_GRAYSCALE);
    cv::Mat out_img{in_img.size(), in_img.type()};
    cv::Mat A{out_img.size(), out_img.type()};
    //---
    for (int i = 0; i < out_img.size().width; i++) {
        for (int j = 0; j < out_img.size().height; j++) {
            if (out_img.at<uchar>(i, j) <= 0) {
                A.at<uchar>(i, j) = (0.0); // a * out_img.at<uchar>(i, j)
            } else {
                A.at<uchar>(i, j) = (out_img.at<uchar>(i, j));
            }
        }
    }

    cv::imwrite("./image/output_3.png", A);
    //---------------------------------
    return _status;
}*/
NET_STATUS NET_Activation(int _layer_num, NET_INFO * _net_info, LAYER_INFO * _layer_info) {
    NET_STATUS _status = NET_STATUS_ERROR;
    //---------------------------------
    float a = 0.1;
    //---
    for (int m = 0; m < _layer_info->nb_of_filters; m++) {
        for (int i = 0; i < _layer_info->dst[0].size().width; i++) {
            for (int j = 0; j < _layer_info->dst[0].size().height; j++) {
                if (_layer_info->dst[m].at<float>(i, j) <= 0) {
                    // _layer_info->dst[m].at<float>(i, j) = (0.0);
                    _layer_info->dst[m].at<float>(i, j) = a * _layer_info->dst[m].at<float>(i, j);
                } else {
                    // _layer_info->dst[m].at<float>(i, j) = a * _layer_info->dst[m].at<float>(i, j);
                }
            }
        }
    }
    //---
    _status = NET_STATUS_OK;
    //---------------------------------
    return _status;
}

NET_STATUS NET_Maxpool(int _layer_num, NET_INFO * _net_info, LAYER_INFO * _layer_info) {
    NET_STATUS _status = NET_STATUS_ERROR;
    //---------------------------------
    int d, i, j, k, l;
    int _in_deep = _layer_infos[_layer_num - 1].nb_of_filters;
    int _width = _layer_infos[_layer_num - 1].dst[0].size().width;
    int _height = _layer_infos[_layer_num - 1].dst[0].size().height;
    cv::Size _matsize;
    int _stride = _layer_info->stride;
    int _kernel_size = _layer_info->kernel_size;
    float _max = 0.0;
    int _prev_layer_index = _layer_num - 1;
    float _me = 0;
    //---
    _matsize.width = _width / _stride;
    _matsize.height = _height / _stride;
    for (d = 0; d < _in_deep; d++) {
        _layer_info->dst[d] = cv::Mat{_matsize, CV_32F};
    }
    //---
    for (d = 0; d < _in_deep; d++) {
        for (i = 0; i < _width; i += _stride) {
            for (j = 0; j < _height; j += _stride) {
                _max = _layer_infos[_prev_layer_index].dst[d].at<float>(i, j);
                for (k = 0; k < _kernel_size; k++) {
                    for (l = 0; l < _kernel_size; l++) {
                        _me = _layer_infos[_prev_layer_index].dst[d].at<float>(i + k, j + l);
                        if (_max < _layer_infos[_prev_layer_index].dst[d].at<float>(i + k, j + l)) {
                            _max = _layer_infos[_prev_layer_index].dst[d].at<float>(i + k, j + l);
                        }
                    }
                }
                _layer_info->dst[d].at<float>(i/2, j/2) = _max;
            }
        }
    }
    //---
    _status = NET_STATUS_OK;
    //---------------------------------
    return _status;
}

NET_STATUS NET_Reorg(int _layer_num, NET_INFO * _net_info, LAYER_INFO * _layer_info) {
    NET_STATUS _status = NET_STATUS_ERROR;
    //---------------------------------
    int d, i, j, k;
    int _offset_row, _offset_col;
    int _in_width, _in_height, _in_deep;
    int _out_width, _out_height, _out_deep;
    int _stride = _layer_info->stride;
    cv::Size _matsize;
    int _prev_layer_num = _layer_num - 1;
    //---
    _in_width = _layer_infos[_prev_layer_num].dst[0].size().width;
    _in_height = _layer_infos[_prev_layer_num].dst[0].size().height;
    _in_deep = _layer_infos[_prev_layer_num].nb_of_filters;
    if (_layer_infos[_prev_layer_num].type == LAYER_TYPE_MAXPOOL){
        _in_deep = _layer_infos[_prev_layer_num - 1].nb_of_filters;
    }
    _out_width = _in_width / _stride;
    _out_height = _in_height / _stride;
    _out_deep = _in_deep * _stride * _stride;
    //---
    _matsize.width = _out_width;
    _matsize.height = _out_height;
    _layer_info->dst.resize(_out_deep);
    for (d = 0; d < _out_deep; d++) {
        _layer_info->dst[d] = cv::Mat{_matsize, CV_32F};
    }
    //---
    for (k = 0; k < _in_deep; k++) {
        for (d = 0; d < _out_deep; d++) {
            //---
            _offset_row = 0;
            if (d > _out_deep / 2) {
                _offset_row = d % _stride;
            }
            //---
            for (i = _offset_row; i < _in_width; i += _stride) {
                _offset_col = d % _stride;
                for (j = _offset_col; j < _in_height; j += _stride) {
                    _layer_info->dst[d].at<float>(i /_stride, j / _stride) = _layer_infos[_prev_layer_num].dst[k].at<float>(i, j);
                }
            }
            //---
        }
    }
    //---
    _status = NET_STATUS_OK;
    //---------------------------------
    return _status;
}

NET_STATUS NET_Route(int _layer_num, NET_INFO * _net_info, LAYER_INFO * _layer_info) {
    NET_STATUS _status = NET_STATUS_ERROR;
    //---------------------------------
    int d;
    int _start_layer = _layer_info->layers[0]; // 26
    int _end_layer = _layer_info->layers[1]; // 24
    cv::Size _matsize;
    int _prev_layer_num = _layer_num - 1;
    int _in_deep = _layer_infos[_prev_layer_num].nb_of_filters;
    //---
    if (_layer_infos[_prev_layer_num].type == LAYER_TYPE_MAXPOOL){
        _in_deep = _layer_infos[_prev_layer_num - 1].nb_of_filters;
    }
    //---
    if (_end_layer == 0) {
        //---
        _layer_infos->dst.resize(_in_deep);
        //---
        _matsize.width = _layer_infos[_start_layer].dst.size();
        _matsize.height = _layer_infos[_start_layer].dst.size();
        //---
        for (d = 0; d < _in_deep; d++) {
            _layer_info->dst[d] = cv::Mat{_matsize, CV_32F};
            _layer_info->dst[d] = _layer_infos[_start_layer].dst[d];
        }
        //---
    } else {
        _layer_infos->dst.resize(_in_deep);
        //---
        _matsize.width = _layer_infos[_start_layer].dst.size();
        _matsize.height = _layer_infos[_start_layer].dst.size();
        //---
        _in_deep = 2048;
        for (d = 0; d < _in_deep; d++) {
            _layer_info->dst[d] = cv::Mat{_matsize, CV_32F};
            _layer_info->dst[d] = _layer_infos[_start_layer].dst[d];
        }
        //---
        _in_deep = 1024;
        for (d = 0; d < _in_deep; d++) {
            _layer_info->dst[d] = cv::Mat{_matsize, CV_32F};
            _layer_info->dst[d] = _layer_infos[_end_layer].dst[d];
        }
        //---
    }
    //---
    _status = NET_STATUS_OK;
    //---------------------------------
    return _status;
}

NET_STATUS NET_Reshape(int _layer_num, NET_INFO * _net_info, LAYER_INFO * _layer_info) {
    NET_STATUS _status = NET_STATUS_ERROR;
    //---------------------------------
    int _width = 13;
    int _height = 13;
    int _in_deep = 425;
    int _nb_of_box = 5;
    int _box_info = 5;
    int _nb_of_classes = 80;
    int _dims = 4;
    int _sz[] = {_width, _height, _nb_of_box, (_box_info + _nb_of_classes)};
    int _idx[] = {0, 0, 0, 0};
    //---
    _layer_info->dst = cv::Mat(_dims, _sz, CV_32F, cv::Scalar(0));
    //---
    for (_idx[0] = 0; _idx[0] < _width; _idx[0]++) {
        for (_idx[1] = 0; _idx[1] < _height; _idx[1]++) {
            for (_idx[2] = 0; _idx[2] < _nb_of_box; _idx[2]++) {
                for (_idx[3] = 0; _idx[3] < (_box_info + _nb_of_classes); _idx[3]++) {
                    _layer_info->dst[0].at<float>(_idx) = _layer_infos[_layer_num].dst[_idx[0]].at<float>(_idx[1], (_nb_of_box + 1) * _idx[3]);
                }
            }
        }
    }
    //---
    //---------------------------------
    return _status;
}

NET_STATUS NET_Sigmoid(int _layer_num, NET_INFO * _net_info, LAYER_INFO * _layer_info) {
    NET_STATUS _status = NET_STATUS_ERROR;
    //---------------------------------
    //---
    // 1 / (1 + np.exp(-x))
    //---------------------------------
    return _status;
}

NET_STATUS NET_Softmax(int _layer_num, NET_INFO * _net_info, LAYER_INFO * _layer_info) {
    NET_STATUS _status = NET_STATUS_ERROR;
    //---------------------------------
    
    //---------------------------------
    return _status;
}

int NET_interval_overlap(float* _interval_a, float* _interval_b) {
    float x1 = _interval_a[0]; 
    float x2 = _interval_a[1];
    float x3 = _interval_b[0];
    float x4 = _interval_b[1];

    if (x3 < x1) {
        if (x4 < x1) {
            return 0;
        } else {
            return min(x2, x4) - x1;
        }
    } else {
        if (x2 < x3) {
             return 0;
        } else {
            return min(x2,x4) - x3;
        }
    }
}

NET_STATUS NET_YoloFunc(int _layer_num, NET_INFO * _net_info, LAYER_INFO * _layer_info) {
    NET_STATUS _status = NET_STATUS_ERROR;
    //---------------------------------
    int i, j, k, c;
    int _width = 13;
    int _height = 13;
    int _in_deep = 425;
    int _nb_of_box = 5;
    int _box_info = 5;
    int _nb_of_classes = 80;
    int _dims = 4;
    int _sz[] = {_width, _height, _nb_of_box, (_box_info + _nb_of_classes)};
    int _idx[] = {0, 0, 0, 0};
    int _idx_2[] = {0, 0, 0, 0};
    float _obj_threshold = 0.3;
    float _max = 0.0;
    float _min = 0.0;
    float _sum = 0;
    float _sigmoid = 0.0;
    float _anchors[] = {0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828};
    std::vector<cv::Mat> _boxes;
    std::vector<cv::Mat> _final_boxes;
    float _nms_threshold = 0.6;
    float _temp_max = 0;
    //---
    NET_Reshape(_layer_num, _net_info, &_layer_infos[_layer_num]);
    //---
    for (_idx[0] = 0; _idx[0] < _width; _idx[0]++) {
        for (_idx[1] = 0; _idx[1] < _height; _idx[1]++) {
            for (_idx[2] = 0; _idx[2] < _nb_of_box; _idx[2]++) {
                for (_idx[3] = _box_info; _idx[3] < _nb_of_box * (_box_info + _nb_of_classes); _idx[3]++) {
                    if ( _layer_infos->dst[0].at<float>(_idx) >= _max) {
                        _max =  _layer_infos->dst[0].at<float>(_idx);
                    }
                    if ( _layer_infos->dst[0].at<float>(_idx) <= _min) {
                        _min =  _layer_infos->dst[0].at<float>(_idx);
                    }
                }
            }
        }
    }
    //---
    for (_idx[0] = 0; _idx[0] < _width; _idx[0]++) {
        for (_idx[1] = 0; _idx[1] < _height; _idx[1]++) {
            for (_idx[2] = 0; _idx[2] < _nb_of_box; _idx[2]++) {
                //---
                _idx[3] = 4;
                _layer_infos->dst[0].at<float>(_idx) = 1 / (1 + exp((-1) * _layer_infos->dst[0].at<float>(_idx)));
                //---
                _sum = 0;
                for (_idx[3] = _box_info; _idx[3] < _nb_of_box * (_box_info + _nb_of_classes); _idx[3]++) {
                    _layer_infos->dst[0].at<float>(_idx) -= _max;
                    _layer_infos->dst[0].at<float>(_idx) = exp(_layer_infos->dst[0].at<float>(_idx));
                    _sum += _layer_infos->dst[0].at<float>(_idx);
                }
                //---
                for (_idx[3] = _box_info; _idx[3] < _nb_of_box * (_box_info + _nb_of_classes); _idx[3]++) {
                    _layer_infos->dst[0].at<float>(_idx) /= _sum;
                }
                //---
                _idx_2[0] = _idx[0];
                _idx_2[1] = _idx[1];
                _idx_2[2] = _idx[2];
                _idx_2[3] = 4;
                for (_idx[3] = _box_info; _idx[3] < _nb_of_box * (_box_info + _nb_of_classes); _idx[3]++) {
                    _layer_infos->dst[0].at<float>(_idx) = _layer_infos->dst[0].at<float>(_idx_2) * _layer_infos->dst[0].at<float>(_idx);
                }
                //---
                for (_idx[3] = _box_info; _idx[3] < _nb_of_box * (_box_info + _nb_of_classes); _idx[3]++) {
                    if (_layer_infos->dst[0].at<float>(_idx) < _obj_threshold) {
                        _layer_infos->dst[0].at<float>(_idx) = 0;
                    }
                }
                //---
                _sum = 0;
                for (_idx[3] = _box_info; _idx[3] < _nb_of_box * (_box_info + _nb_of_classes); _idx[3]++) {
                    _sum += _layer_infos->dst[0].at<float>(_idx);
                }
                //---
                if (_sum > 0) {
                    _idx[3] = 0;
                    _sigmoid = 1 / (1 + exp((-1) * _layer_infos->dst[0].at<float>(_idx)));
                    _layer_infos->dst[0].at<float>(_idx) = (_idx[0] + _sigmoid) / _width;

                    _idx[3] = 1;
                    _sigmoid = 1 / (1 + exp((-1) * _layer_infos->dst[0].at<float>(_idx)));
                    _layer_infos->dst[0].at<float>(_idx) = (_idx[1] + _sigmoid) / _height;

                    _idx[3] = 2;
                    _layer_infos->dst[0].at<float>(_idx) = _anchors[2 * _idx[2] + 0] * exp(_layer_infos->dst[0].at<float>(_idx)) / _width;
                    _idx[3] = 3;
                    _layer_infos->dst[0].at<float>(_idx) = _anchors[2 * _idx[2] + 1] * exp(_layer_infos->dst[0].at<float>(_idx)) / _height;

                    _boxes.push_back(_layer_infos->dst[0].at<cv::Mat>(_idx[0], _idx[1], _idx[2]));
                }
                //---
            }
        }
    }
    //---
    for (c = 0; c < _nb_of_classes; c++) {
        std::vector<int> _sorted_indices;
        //--- sorting -----------------
        //-----------------------------
        for (i = 0; i < _sorted_indices.size(); i++) {
            int _index_i = _sorted_indices[i];

            if (_boxes[_index_i].at<float>(_box_info + c) == 0) {
                continue;
            } else {
                for (j = i + 1; j < _sorted_indices.size(); j++) {
                    int _index_j = _sorted_indices[j];
                    float _my_score = 0;
                    float _box1_xmax, _box1_xmin, _box1_ymax, _box1_ymin;
                    float _box2_xmax, _box2_xmin, _box2_ymax, _box2_ymin;

                    _box1_xmax = _boxes[_index_i].at<float>(0) + _width / 2;
                    _box1_xmin = _boxes[_index_i].at<float>(0) - _width / 2;
                    _box1_ymax = _boxes[_index_i].at<float>(1) + _height / 2;
                    _box1_ymin = _boxes[_index_i].at<float>(1) - _height / 2;
                    
                    _box2_xmax = _boxes[_index_j].at<float>(0) + _width / 2;
                    _box2_xmin = _boxes[_index_j].at<float>(0) - _width / 2;
                    _box2_ymax = _boxes[_index_j].at<float>(1) + _height / 2;
                    _box2_ymin = _boxes[_index_j].at<float>(1) - _height / 2;

                    float _box1x[] ={_box1_xmin, _box1_xmax};
                    float _box2x[] ={_box2_xmin, _box2_xmax};
                    float _box1y[] ={_box1_ymin, _box1_ymax};
                    float _box2y[] ={_box2_ymin, _box2_ymax};

                    float _intersect_w = NET_interval_overlap(_box1x, _box2x);
                    float _intersect_h = NET_interval_overlap(_box1y, _box2y);
                    
                    float _intersect = _intersect_w * _intersect_h;

                    float _w1 = _box1_xmax-_box1_xmin;
                    float _h1 = _box1_ymax-_box1_ymin;
                    float _w2 = _box2_xmax-_box2_xmin;
                    float _h2 = _box2_ymax-_box2_ymin;
                    
                    float _union = (_w1 * _h1) + (_w2 * _h2) - _intersect;
                    
                    _my_score = float(_intersect) / _union;

                    if (_my_score >= _nms_threshold) {
                        _boxes[_index_j].at<float>(_box_info + c) = 0;
                    }
                }
            }
        }

    }
    //---
    for (i = 0; i < _boxes.size(); i++) {
        //---
        _temp_max = 0;
        for (j = _box_info; j < _box_info + _nb_of_classes; j++) {
            if (_boxes[i].at<float>(j) >= _temp_max) {
                _temp_max = _boxes[i].at<float>(j);
                k = j;
            }
        }
        //---
        if (_boxes[i].at<float>(k) > _obj_threshold) {
            _final_boxes.push_back(_boxes[i]);
        }
        //---
    }
    //---------------------------------
    return _status;
}

NET_STATUS NET_Do(const cv::String &_img_name) {
    NET_STATUS _status = NET_STATUS_ERROR;
    //---------------------------------
    NET_INFO _net_info;
    ofstream _myfile;
    char _file_path[200] = {0};
    char _str_m[10] = {0};
    //---
    for (int i = 0 ; i < _nb_of_layer; i++) {
        switch (_layer_infos[i].type)
        {
            case LAYER_TYPE_INPUT:{
                int d;
                int _in_deep = 3;
                cv::Size _matsize;
                cv::Mat _input_img;
                //---
                _matsize.width = 416;
                _matsize.height = 416;
                _layer_infos[i].dst.resize(_in_deep);
                for (d = 0; d < _in_deep; d++) {
                    _layer_infos[i].dst[d] = cv::Mat{_matsize, CV_32F};;
                }
                //---
                _input_img = cv::imread(_img_name, cv::IMREAD_COLOR); // IMREAD_GRAYSCALE
                cv::split(_input_img, _layer_infos[i].dst);
                for (d = 0; d < _in_deep; d++) {
                    _layer_infos[i].dst[d].convertTo(_layer_infos[i].dst[d], CV_32F);
                    _layer_infos[i].dst[d] /= 255;
                }
                //---
                break;
            }
            case LAYER_TYPE_CONV: {
                //---
                conv_layer_counter ++;
                norm_layer_counter ++;
                leaky_re_lu_layer_counter ++;
                //---
                _layer_infos[i].dst.resize(_layer_infos[i].nb_of_filters);
                _status = NET_Padding(i, &_net_info, &_layer_infos[i]);
                _status = NET_Conv(i, &_net_info, &_layer_infos[i]);
                _status = NET_BatchNorm(i, &_net_info, &_layer_infos[i]);
                _status = NET_Activation(i, &_net_info, &_layer_infos[i]);
                //---
                for (int m = 0; m < _layer_infos[i].nb_of_filters; m++) {
                    sprintf(_file_path, "./outputs/leaky_re_lu_");
                    if (leaky_re_lu_layer_counter != 0) {
                        sprintf(_str_m, "%d", leaky_re_lu_layer_counter);
                        strcat(_file_path, _str_m);
                    }
                    if (leaky_re_lu_layer_counter != 0) {
                        strcat(_file_path, "_d_");
                    } else {
                        strcat(_file_path, "d_");
                    }
                    sprintf(_str_m, "%d", m);
                    strcat(_file_path, _str_m);
                    strcat(_file_path, ".bin");
                    _myfile.open(_file_path, ios::out | ios::binary);
                    if (_myfile.is_open()) {
                        _myfile.write(reinterpret_cast<char *>(_layer_infos[i].dst[m].data), _layer_infos[i].dst[m].size().width * _layer_infos[i].dst[m].size().height * sizeof(float));
                        _myfile.close();
                    }
                    else {
                        cout << "Unable to open file!";
                    }
                }
                //---
                break;
            }
            case LAYER_TYPE_MAXPOOL: {
                //---
                max_pooling2d_layer_counter ++;
                //---
                _layer_infos[i].dst.resize(_layer_infos[i - 1].nb_of_filters);
                _status = NET_Maxpool(i, &_net_info, &_layer_infos[i]);
                //---
                for (int m = 0; m < _layer_infos[i - 1].nb_of_filters; m++) {
                    sprintf(_file_path, "./outputs/max_pooling2d_");
                    if (max_pooling2d_layer_counter != 0) {
                        sprintf(_str_m, "%d", max_pooling2d_layer_counter);
                        strcat(_file_path, _str_m);
                    }
                    if (max_pooling2d_layer_counter != 0) {
                        strcat(_file_path, "_d_");
                    } else {
                        strcat(_file_path, "d_");
                    }
                    sprintf(_str_m, "%d", m);
                    strcat(_file_path, _str_m);
                    strcat(_file_path, ".bin");
                    _myfile.open(_file_path, ios::out | ios::binary);
                    if (_myfile.is_open()) {
                        _myfile.write(reinterpret_cast<char *>(_layer_infos[i].dst[m].data), _layer_infos[i].dst[m].size().width * _layer_infos[i].dst[m].size().height * sizeof(float));
                        _myfile.close();
                    }
                    else {
                        cout << "Unable to open file!";
                    }
                }
                //---
                break;
            }
            case LAYER_TYPE_REORG: {
                _status = NET_Reorg(i, &_net_info, &_layer_infos[i]);
                break;
            }
            case LAYER_TYPE_ROUTE: {
                _layer_infos[i].layers[0] += i;
                if (_layer_infos[i].layers[1] != 0) {
                    _layer_infos[i].layers[1] += i;
                }

                _status = NET_Route(i, &_net_info, &_layer_infos[i]);
                break;
            }
            case LAYER_TYPE_YOLO: {
                NET_YoloFunc(i, &_net_info, &_layer_infos[i]);
                break;
            }
            default: {
            break;
        }
        }
    }
    //---------------------------------
    return _status;
}

int main()
{
    //---------------------------------
    vector<string> msg {"Hello", "C++", "World", "from", "VS Code", "and the C++ extension!"};

    for (const string& word : msg)
    {
        cout << word << " ";
    }
    cout << endl;
    //---------------------------------
    NET_CreateModel();
    //---------------------------------
    NET_LoadWeights();
    //---------------------------------
    // NET_Do("./image/lenna.png");
    NET_Do("./image/my_test.jpg");
    //---------------------------------
    //--- test ---
    //---------------------------------
    //--- filter ---
    //---------------------------------
    /*cv::Mat src, dst;

    float sobelx_data[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    // float sobelx_data[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
    cv::Mat sobelx = cv::Mat(3, 3, CV_32F, sobelx_data);
    std::cout << "sobelx = " << std::endl << sobelx << std::endl;

    src = cv::imread("./image/lenna.png", cv::IMREAD_GRAYSCALE);
    cv::filter2D(src, dst, -1, sobelx, cv::Point(-1, -1), 0, cv::BORDER_DEFAULT);
    cv::imwrite("./image/output.png", dst);
    //---------------------------------
    //--- convolution ---
    //---------------------------------
    int i, j, k, l;
    uchar sum;
    int stride = 1;
    cv::Mat in_img;

    // float kernel_data[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    uchar kernel_data[9] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
    cv::Mat kernel = cv::Mat(3, 3, CV_8U, kernel_data);
    std::cout << "kernel = " << std::endl << kernel << std::endl;

    in_img = cv::imread("./image/lenna.png", cv::IMREAD_GRAYSCALE);
    uchar me = 0;
    uchar me_1 = 0;
    uchar me_2 = 0;
    
    cv::Mat out_img{in_img.size(), in_img.type()};
    
    for (i = 0; i < in_img.size().width; i+=stride) {
        for (j = 0; j < in_img.size().height; j+=stride) {
            sum = 0;
            for (k = -1; k <= 1; k++) {
                for (l = -1; l <= 1; l++) {
                    if (((i + k) >= 0 && (i + k) < in_img.size().width) &&
                        ((j + l) >= 0 && (j + l) < in_img.size().width))
                    {
                        me_1 = in_img.at<uchar>(i + k, j + l);
                        me_2 = kernel.at<uchar>(k + 1, l + 1);
                        sum += me_1 * me_2;
                    }
                }
            }
            out_img.at<uchar>(i, j) = (uchar)sum;
        }
    }

    cv::imwrite("./image/output_2.png", out_img);
    //---------------------------------
    //--- activation ReLU ---
    //---------------------------------
    cv::Mat A{out_img.size(), out_img.type()};
    // int a = 1;
    
    for (int i = 0; i < out_img.size().width; i++) {
        for (int j = 0; j < out_img.size().height; j++) {
            if (out_img.at<uchar>(i, j) <= 0) {
                A.at<uchar>(i, j) = (0.0); // a * out_img.at<uchar>(i, j)
            } else {
                A.at<uchar>(i, j) = (out_img.at<uchar>(i, j));
            }
        }
    }

    cv::imwrite("./image/output_3.png", A);
    //---------------------------------
    //--- padding ---
    //---------------------------------
    cv::Mat borderd_img;
    cv::Scalar value = 0;

    cv::copyMakeBorder(in_img, borderd_img, 3, 3, 3, 3, cv::BORDER_CONSTANT, value);
    
    cv::imwrite("./image/output_4.png", borderd_img);*/
    //---------------------------------
    return 0;
    //---------------------------------
}
