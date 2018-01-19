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

//https://stackoverflow.com/questions/981378/how-to-recognize-vehicle-license-number-plate-anpr-from-an-image/37523538#37523538

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
using namespace cv;
using namespace std;


#define LOCATION_METERAN_AIR "data/img/meteran_air.jpg"
#define MAX_IMAGE_SIZE 500
const string LOCATION_SAVED = "data/saved/";
const string TYPE_SAVED = ".png";

#define LOCATION_TRAIN_DIGIT_IMAGE "data/trainDataDigit/digits.png"
#define LOCATION_SAVED_MODEL_SVM "data/trainDataDigit/eyeGlassClassifierModel.yml"

Mat src;
Mat src_gray;
Mat imageBlokAngkaExtracted;
vector<Mat> listImageAngkaExtracted;
RNG rng(12345);

#include "blokAngkaExtraction.h"
#include "angkaExtraction.h"
#include "angkaRecognition.h"


/// Function header
void run_program();

/** @function main */
int main( int argc, char** argv )
{
    /// Load source image and convert it to gray
    src = imread( LOCATION_METERAN_AIR, 1 );
    //Kecilin gambar kalo kegedean (agar nantinya mengecilkan waktu komputasi)


    if(src.rows>=MAX_IMAGE_SIZE) {
        float ratio_new_image = src.rows / MAX_IMAGE_SIZE;
        int newImageWidth = src.cols / ratio_new_image;
        int newImageHeight = src.rows / ratio_new_image;
        resize(src, src, Size(newImageWidth,newImageHeight));
    }

    /// Show original image
    //namedWindow( "Source", CV_WINDOW_AUTOSIZE );
    //imshow( "Source", src );


    blokAngkaExtractionInit();
    angkaExtractionInit();
    angkaRecognitionInit();

    run_program();

    waitKey(0);
    return(0);
}

void run_program()
{
    blokAngkaExtraction(0,0);
    angkaExtraction(0,0);
    angkaRecognition(0,0);
}
