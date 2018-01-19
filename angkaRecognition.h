/*
*
* Copyright (C) 2018 Albert Alfrianta @TOMCAT Inc.
*
* feel free to modify this program, under the GNU License :)
* give your feedback if you love this program or want to give a contribution
* ask me if you have any question
*
* January 2018
* albert.brucelee@gmail.com
* https://github.com/albertbrucelee/MeteranAirRecognition
*
* SPREAD OUT THE POWER OF OPEN SOURCE!
*
*/

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
using namespace cv;
using namespace std;


#include "svm.h"


const string TITLE_ANGKA_RECOGNITION_PREPROCESS_RESULT = "Angka-Recognition-Preprocess";

Ptr<SVM> model;

Mat preprocessRecognition(Mat img);

void angkaRecognitionInit() {
    model = svmClassification(LOCATION_TRAIN_DIGIT_IMAGE);
}

void angkaRecognition(int, void*) {
    cout << "Recognition Result" << endl;
    for (int i=0; i<listImageAngkaExtracted.size(); i++) {
        Mat angkaImg = preprocessRecognition(listImageAngkaExtracted[i]);
        imwrite( LOCATION_SAVED+TITLE_ANGKA_RECOGNITION_PREPROCESS_RESULT+"_"+to_string(i)+TYPE_SAVED, angkaImg );

        float number = testClassify(model, angkaImg);
        cout << (int)number << " ";
    }
    cout << endl;

}


Mat preprocessRecognition(Mat img)
{
    /// ########## 1.2 Center the bounding box's contents ########## ///

    Mat newImg;
    double ratioNewImg = 0.6;
    int newImgRowsInc = img.rows*ratioNewImg;
    int newImgColsInc = img.cols*ratioNewImg;


    //make it even number
    if(newImgRowsInc%2==1) {
        newImgRowsInc++;
    }
    if(newImgColsInc%2==1) {
        newImgColsInc++;
    }

    int newImgRows = img.rows + newImgRowsInc;
    int newImgCols = img.cols + newImgColsInc;

    int startAtX = 0 + newImgColsInc/2;
    int startAtY = 0 + newImgRowsInc/2;


    //make it square
    if(newImgRows<newImgCols) {
        startAtY += (newImgCols-newImgRows)/2;
        newImgRows = newImgCols;
    } else {
        startAtX += (newImgRows-newImgCols)/2;
        newImgCols = newImgRows;
    }

    newImg = newImg.zeros(newImgRows, newImgCols, CV_8UC1);
    //int startAtX = (newImg.cols/2)-(colRight-colLeft)/2;

    //int startAtY = (newImg.rows/2)-(rowBottom-rowTop)/2;

    for(int y=0; y<img.rows; y++)
    {
        uchar *ptr = newImg.ptr<uchar>(y+startAtY);
        for(int x=0; x<img.cols; x++)
        {
            ptr[x+startAtX] = img.at<uchar>(y,x);
        }
    }

    return newImg;

}


