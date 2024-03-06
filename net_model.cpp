/*** Includes ******************************************************************
*/
#include <stdbool.h>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <string>

#include "net_model.h"

using namespace std;
/*******************************************************************************
*/

/*** defines *******************************************************************
*/
/*******************************************************************************
*/

/*** typedef *******************************************************************
*/

typedef enum layer_type {
    LAYER_TYPE_ERROR = 0,
    LAYER_TYPE_CONV,
    LAYER_TYPE_MAXPOOL,
    LAYER_TYPE_ROUTE,
    LAYER_TYPE_REORG
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
    uint8_t nb_of_filters;
    uint8_t kernel_size;
    uint8_t stride;
    uint8_t pad;
    uint8_t batch_normalize;
    ACTIVATION_TYPE activation;
    // const LAYER_INFO * next;
} LAYER_INFO;

typedef struct net_model {
    const char * name;
    const char * desc;
    uint8_t nb_of_layers;
    const LAYER_INFO * first_layer;
} NET_MODEL;


typedef struct layer_type_mapping {
    string str;
    LAYER_TYPE type;
} LAYER_TYPE_MAPPING;

/*******************************************************************************
*/

/*** const *********************************************************************
*/

//--- layers ---
/*
static const LAYER_INFO layer_1 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 32,
    .kernel_size = 3,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_2,
};

static const LAYER_INFO layer_2 = {
    .type = LAYER_TYPE_MAXPOOL,
    .nb_of_filters = 0,
    .kernel_size = 2,
    .stride = 2,
    .pad = 0,
    .batch_normalize = 0,
    .activation = ACTIVATION_TYPE_ERROR,
    .next = &layer_3,
};

static const LAYER_INFO layer_3 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 64,
    .kernel_size = 3,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_4,
};

static const LAYER_INFO layer_4 = {
    .type = LAYER_TYPE_MAXPOOL,
    .nb_of_filters = 0,
    .kernel_size = 2,
    .stride = 2,
    .pad = 0,
    .batch_normalize = 0,
    .activation = ACTIVATION_TYPE_ERROR,
    .next = &layer_5,
};

static const LAYER_INFO layer_5 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 128,
    .kernel_size = 3,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_6,
};

static const LAYER_INFO layer_6 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 64,
    .kernel_size = 1,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_7,
};

static const LAYER_INFO layer_7 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 128,
    .kernel_size = 3,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_8,
};

static const LAYER_INFO layer_8 = {
    .type = LAYER_TYPE_MAXPOOL,
    .nb_of_filters = 0,
    .kernel_size = 2,
    .stride = 2,
    .pad = 0,
    .batch_normalize = 0,
    .activation = ACTIVATION_TYPE_ERROR,
    .next = &layer_9,
};

static const LAYER_INFO layer_9 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 256,
    .kernel_size = 3,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_10,
};

static const LAYER_INFO layer_10 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 128,
    .kernel_size = 1,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_11,
};

static const LAYER_INFO layer_11 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 256,
    .kernel_size = 3,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_12,
};

static const LAYER_INFO layer_12 = {
    .type = LAYER_TYPE_MAXPOOL,
    .nb_of_filters = 0,
    .kernel_size = 2,
    .stride = 2,
    .pad = 0,
    .batch_normalize = 0,
    .activation = ACTIVATION_TYPE_ERROR,
    .next = &layer_13,
};

static const LAYER_INFO layer_13 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 512,
    .kernel_size = 3,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_14,
};

static const LAYER_INFO layer_14 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 256,
    .kernel_size = 1,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_15,
};

static const LAYER_INFO layer_15 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 512,
    .kernel_size = 3,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_16,
};

static const LAYER_INFO layer_16 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 256,
    .kernel_size = 1,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_17,
};

static const LAYER_INFO layer_17 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 512,
    .kernel_size = 3,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_18,
};

static const LAYER_INFO layer_18 = {
    .type = LAYER_TYPE_MAXPOOL,
    .nb_of_filters = 0,
    .kernel_size = 2,
    .stride = 2,
    .pad = 0,
    .batch_normalize = 0,
    .activation = ACTIVATION_TYPE_ERROR,
    .next = &layer_19,
};

static const LAYER_INFO layer_19 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 1024,
    .kernel_size = 3,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_20,
};

static const LAYER_INFO layer_20 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 512,
    .kernel_size = 1,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_21,
};

static const LAYER_INFO layer_21 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 1024,
    .kernel_size = 3,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_22,
};

static const LAYER_INFO layer_22 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 512,
    .kernel_size = 1,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_23,
};

static const LAYER_INFO layer_23 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 1024,
    .kernel_size = 3,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_24,
};

static const LAYER_INFO layer_24 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 1024,
    .kernel_size = 3,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_25,
};

static const LAYER_INFO layer_25 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 1024,
    .kernel_size = 3,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_26,
};

static const LAYER_INFO layer_26 = {
    .type = LAYER_TYPE_ROUTE,
    .layers = {-9, 0, 0},
    .nb_of_filters = 0,
    .kernel_size = 0,
    .stride = 0,
    .pad = 0,
    .batch_normalize = 0,
    .activation = ACTIVATION_TYPE_ERROR,
    .next = &layer_27,
};

static const LAYER_INFO layer_27 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 64,
    .kernel_size = 1,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_28,
};

static const LAYER_INFO layer_28 = {
    .type = LAYER_TYPE_REORG,
    .nb_of_filters = 0,
    .kernel_size = 0,
    .stride = 2,
    .pad = 0,
    .batch_normalize = 0,
    .activation = ACTIVATION_TYPE_ERROR,
    .next = &layer_29,
};

static const LAYER_INFO layer_29 = {
    .type = LAYER_TYPE_ROUTE,
    .layers = {-1, -4, 0},
    .nb_of_filters = 0,
    .kernel_size = 0,
    .stride = 0,
    .pad = 0,
    .batch_normalize = 0,
    .activation = ACTIVATION_TYPE_ERROR,
    .next = &layer_30,
};

static const LAYER_INFO layer_30 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 1024,
    .kernel_size = 3,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 1,
    .activation = ACTIVATION_TYPE_LEAKY,
    .next = &layer_31,
};

static const LAYER_INFO layer_31 = {
    .type = LAYER_TYPE_CONV,
    .nb_of_filters = 425,
    .kernel_size = 1,
    .stride = 1,
    .pad = 1,
    .batch_normalize = 0,
    .activation = ACTIVATION_TYPE_LINEAR,
    .next = NULL,
};
*/

//--- model ---
/*const static NET_MODEL net_model = {
    .name = "yolov2",
    .desc = "this is desc",
    .nb_of_layers = 31,
    .first_layer = &layer_1
};*/

/*******************************************************************************
*/

/*** Private Function Definition ***********************************************
*/
// static NET_STATUS NET_Constructor(NET_MODEL * _model);
// static NET_STATUS NET_Destructor(NET_MODEL * _model);
/*******************************************************************************
*/

/*** Global var ****************************************************************
*/
/*******************************************************************************
*/

/*** extern from other sources *************************************************
*/
/*******************************************************************************
*/

/*** Function Declaration ******************************************************
*/

/*static NET_STATUS NET_Constructor(NET_MODEL * _model) {
    NET_STATUS _status = NET_STATUS_ERROR;
    //---------------------------------
    // _model = (NET_MODEL *)malloc(sizeof(NET_MODEL));
    //---------------------------------
    return _status;
}

static NET_STATUS NET_Destructor(NET_MODEL * _model) {
    NET_STATUS _status = NET_STATUS_ERROR;
    //---------------------------------
    // delete(_model);
    //---------------------------------
    return _status;
}*/

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
    int _counter = 0;
    uint8_t _nb_of_layer = 31;
    LAYER_INFO _layer_infos[31];
    LAYER_TYPE_MAPPING _layer_type_mapping[] = {
        {"convolutional", LAYER_TYPE_CONV   },
        {"maxpool"      , LAYER_TYPE_MAXPOOL},
        {"route"        , LAYER_TYPE_ROUTE  },
        {"reorg"        , LAYER_TYPE_REORG  },
    };
    //---
    _myfile.open("yolov2.cfg");
    
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
                cout << "\n" << "[";
                _pos = _line.find("]");
                _pos--;
                cout << "\n" << _pos;
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

/*******************************************************************************
*/

/***************************** End Of File *************************************
*/
