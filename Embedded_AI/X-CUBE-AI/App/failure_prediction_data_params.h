/**
  ******************************************************************************
  * @file    failure_prediction_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-03-13T11:08:28+0100
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef FAILURE_PREDICTION_DATA_PARAMS_H
#define FAILURE_PREDICTION_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_FAILURE_PREDICTION_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_failure_prediction_data_weights_params[1]))
*/

#define AI_FAILURE_PREDICTION_DATA_CONFIG               (NULL)


#define AI_FAILURE_PREDICTION_DATA_ACTIVATIONS_SIZES \
  { 1536, }
#define AI_FAILURE_PREDICTION_DATA_ACTIVATIONS_SIZE     (1536)
#define AI_FAILURE_PREDICTION_DATA_ACTIVATIONS_COUNT    (1)
#define AI_FAILURE_PREDICTION_DATA_ACTIVATION_1_SIZE    (1536)



#define AI_FAILURE_PREDICTION_DATA_WEIGHTS_SIZES \
  { 24532, }
#define AI_FAILURE_PREDICTION_DATA_WEIGHTS_SIZE         (24532)
#define AI_FAILURE_PREDICTION_DATA_WEIGHTS_COUNT        (1)
#define AI_FAILURE_PREDICTION_DATA_WEIGHT_1_SIZE        (24532)



#define AI_FAILURE_PREDICTION_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_failure_prediction_activations_table[1])

extern ai_handle g_failure_prediction_activations_table[1 + 2];



#define AI_FAILURE_PREDICTION_DATA_WEIGHTS_TABLE_GET() \
  (&g_failure_prediction_weights_table[1])

extern ai_handle g_failure_prediction_weights_table[1 + 2];


#endif    /* FAILURE_PREDICTION_DATA_PARAMS_H */
