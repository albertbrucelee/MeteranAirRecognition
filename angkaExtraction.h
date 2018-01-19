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

const string TITLE_ANGKA_HITAM_PREPOCESS_TO_GRAY = "Angka_Hitam-Preprocess-to_gray";
const string TITLE_ANGKA_HITAM_PREPOCESS_THRESHOLD = "Angka_Hitam-Preprocess-threshold";
const string TITLE_ANGKA_HITAM_PREPOCESS_DILATE = "Angka_Hitam-Preprocess-dilate";
const string TITLE_ANGKA_HITAM_FEATURE_EXTRACTION = "Angka_Hitam-FeatureExtraction";
const string TITLE_ANGKA_HITAM_FEATURE_EXTRACTION_RESULT = "Angka_Hitam-FeatureExtraction-result";

const string TITLE_ANGKA_MERAH_PREPOCESS_TO_HSV = "Angka_Merah-Preprocess-to_hsv";
const string TITLE_ANGKA_MERAH_PREPOCESS_HUE_LOWER_RED = "Angka_Merah-Preprocess-hue_lower_red";
const string TITLE_ANGKA_MERAH_PREPOCESS_HUE_UPPER_RED = "Angka_Merah-Preprocess-hue_upper_red";
const string TITLE_ANGKA_MERAH_PREPOCESS_JOIN_HUE_LOWER_UPPER_RED = "Angka_Merah-Preprocess-join_hue_lower_upper_red";
const string TITLE_ANGKA_MERAH_PREPOCESS_DILATE = "Angka_Merah-Preprocess-dilate";
const string TITLE_ANGKA_MERAH_FEATURE_EXTRACTION = "Angka_Merah-FeatureExtraction";
const string TITLE_ANGKA_MERAH_FEATURE_EXTRACTION_RESULT = "Angka_Merah-FeatureExtraction-result";

const string TITLE_ANGKA_FEATURE_EXTRACTION_THRESHOLD = "-threshold";
const string TITLE_ANGKA_FEATURE_EXTRACTION_CONTOURS = "-contours";
const string TITLE_ANGKA_FEATURE_EXTRACTION_CONTOURS_INTEREST = "-contours_interest";



const string TRACKBAR_WINDOW_ANGKA = "Trackbar angkaExtraction";

int value_angkaExtraction_preprocess_blurKernel = 5;
int max_value_angkaExtraction_preprocess_blurKernel = 10;

int value_angkaExtraction_preprocess_sobelKernel = 3;
int max_value_angkaExtraction_preprocess_sobelKernel = 31;

int value_angkaExtraction_preprocess_morphological_width = 16;
int max_value_angkaExtraction_preprocess_morphological_width = 20;

int value_angkaExtraction_preprocess_morphological_height = 4;
int max_value_angkaExtraction_preprocess_morphological_height = 20;

int value_angkaExtraction_featureExtraction_thresh = 46;
int max_value_angkaExtraction_featureExtraction_thresh = 255;

int value_angkaExtraction_featureExtraction_ratioRectangleLengthWidth = 22;     //cannot double, so make int, then divide by 10
int max_value_angkaExtraction_featureExtraction_ratioRectangleLengthWidth = 50;


void angkaExtractionInit();
void angkaExtraction(int, void*);
Mat angkaExtraction_preprocess(Mat srcImg);
void angkaExtraction_preprocess2();
Mat angkaExtraction_preprocessAngkaMerah(Mat srcImg);
vector<Mat> angkaExtraction_featureExtraction(Mat img, string interestWindowName);
void getInterestAngka(double ratioXY, vector<RotatedRect> rect, vector<Vec4i> hierarchy, vector<RotatedRect> *interestImage_point_src, Size sizeBlokAngka);
void drawInterestAngka(vector<RotatedRect> rect, String windowName, Size windowSize);

void angkaExtractionInit() {

    namedWindow( TRACKBAR_WINDOW_ANGKA, CV_WINDOW_AUTOSIZE );

    createTrackbar( " Blur:", TRACKBAR_WINDOW_ANGKA, &value_angkaExtraction_preprocess_blurKernel, max_value_angkaExtraction_preprocess_blurKernel, angkaExtraction );
    createTrackbar( " Sobel:", TRACKBAR_WINDOW_ANGKA, &value_angkaExtraction_preprocess_sobelKernel, max_value_angkaExtraction_preprocess_sobelKernel, angkaExtraction );
    createTrackbar( " Morphological Width:", TRACKBAR_WINDOW_ANGKA, &value_angkaExtraction_preprocess_morphological_width, max_value_angkaExtraction_preprocess_morphological_width, angkaExtraction );
    createTrackbar( " Morphological Height:", TRACKBAR_WINDOW_ANGKA, &value_angkaExtraction_preprocess_morphological_height, max_value_angkaExtraction_preprocess_morphological_width, angkaExtraction );
    createTrackbar( " Threshold:", TRACKBAR_WINDOW_ANGKA, &value_angkaExtraction_featureExtraction_thresh, max_value_angkaExtraction_featureExtraction_thresh, angkaExtraction );
    createTrackbar( " Ratio Rectangle Length Width:", TRACKBAR_WINDOW_ANGKA, &value_angkaExtraction_featureExtraction_ratioRectangleLengthWidth, max_value_angkaExtraction_featureExtraction_ratioRectangleLengthWidth, angkaExtraction );

}

void angkaExtraction(int, void*) {
    Mat img;
    vector<Mat> listImgExtracted;

    ///preproses angka hitam
    img = angkaExtraction_preprocess(imageBlokAngkaExtracted);

    ///extract angka hitam
    listImgExtracted = angkaExtraction_featureExtraction(img, TITLE_ANGKA_HITAM_FEATURE_EXTRACTION);
    listImageAngkaExtracted = listImgExtracted;
    for(int i=0; i<listImgExtracted.size(); i++) {
        imwrite( LOCATION_SAVED+TITLE_ANGKA_HITAM_FEATURE_EXTRACTION_RESULT+"_"+to_string(i)+TYPE_SAVED, listImgExtracted[i] );
    }

    ///preproses angka merah
    img = angkaExtraction_preprocessAngkaMerah(imageBlokAngkaExtracted);

    ///extract angka merah
    listImgExtracted = angkaExtraction_featureExtraction(img, TITLE_ANGKA_MERAH_FEATURE_EXTRACTION);
    listImageAngkaExtracted.insert(listImageAngkaExtracted.end(), listImgExtracted.begin(), listImgExtracted.end());
    for(int i=0; i<listImgExtracted.size(); i++) {
        imwrite( LOCATION_SAVED+TITLE_ANGKA_MERAH_FEATURE_EXTRACTION_RESULT+"_"+to_string(i)+TYPE_SAVED, listImgExtracted[i] );
    }
}


Mat angkaExtraction_preprocess(Mat srcImg) {
    Mat img;
    /// Convert image to gray
    cvtColor( srcImg, img, CV_BGR2GRAY );
    imshow( TITLE_ANGKA_HITAM_PREPOCESS_TO_GRAY, img );
    imwrite( LOCATION_SAVED+TITLE_ANGKA_HITAM_PREPOCESS_TO_GRAY+TYPE_SAVED, img );

    /*
    /// blur untuk menghilangkan noise
    if (value_blokAngkaExtraction_preprocess_blurKernel%2==0) {
        return;
    }
    blur( img, img, Size(value_blokAngkaExtraction_preprocess_blurKernel,value_blokAngkaExtraction_preprocess_blurKernel) );
    imshow( "angkaExtraction_preprocess - Enhancement - Blur", img );
    */

    //Mat img = Mat(img.size(), CV_8UC1);
    inRange(img, Scalar(0), Scalar(50), img);
    //adaptiveThreshold(img, img, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 101, 1);
    imshow(TITLE_ANGKA_HITAM_PREPOCESS_THRESHOLD, img);
    imwrite( LOCATION_SAVED+TITLE_ANGKA_HITAM_PREPOCESS_THRESHOLD+TYPE_SAVED, img );

    //int ratioNewImage = 3;
    //int newImage_row = img.rows * ratioNewImage;
    //int newImage_col = img.cols * ratioNewImage;
    //resize(img,img, Size(newImage_col, newImage_row));
    //imshow("angkaExtraction_preprocess - resize", img);


    Mat kernel = (Mat_<uchar>(5,5) << 0,1,0,1,1,1,0,1,0);
    //erode(img, img, kernel);
    //imshow("erode", img);

    kernel = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
    dilate(img, img, kernel);
    imshow(TITLE_ANGKA_HITAM_PREPOCESS_DILATE, img);
    imwrite( LOCATION_SAVED+TITLE_ANGKA_HITAM_PREPOCESS_DILATE+TYPE_SAVED, img );

    return img;

}

Mat angkaExtraction_preprocessAngkaMerah(Mat srcImg) {
    Mat img = Mat(srcImg.size(), CV_8UC1);
    cvtColor(srcImg, img, COLOR_BGR2HSV);
    imshow( TITLE_ANGKA_MERAH_PREPOCESS_TO_HSV, img );
    imwrite( LOCATION_SAVED+TITLE_ANGKA_MERAH_PREPOCESS_TO_HSV+TYPE_SAVED, img );

    Mat lower_red_hue_range;
    inRange(img, Scalar(0, 70, 70), Scalar(10, 255, 255), lower_red_hue_range);
    imshow( TITLE_ANGKA_MERAH_PREPOCESS_HUE_LOWER_RED, lower_red_hue_range );
    imwrite( LOCATION_SAVED+TITLE_ANGKA_MERAH_PREPOCESS_HUE_LOWER_RED+TYPE_SAVED, lower_red_hue_range );

    Mat upper_red_hue_range;
    inRange(img, Scalar(140, 100, 100), Scalar(200, 255, 255), upper_red_hue_range);
    imshow( TITLE_ANGKA_MERAH_PREPOCESS_HUE_UPPER_RED, upper_red_hue_range );
    imwrite( LOCATION_SAVED+TITLE_ANGKA_MERAH_PREPOCESS_HUE_UPPER_RED+TYPE_SAVED, upper_red_hue_range );

    Mat join_lower_upper_red_hue = Mat(srcImg.size(), CV_8UC1);
    addWeighted(lower_red_hue_range, 1.0, upper_red_hue_range, 1.0, 0.0, join_lower_upper_red_hue);
    imshow(TITLE_ANGKA_MERAH_PREPOCESS_JOIN_HUE_LOWER_UPPER_RED, join_lower_upper_red_hue);
    imwrite( LOCATION_SAVED+TITLE_ANGKA_MERAH_PREPOCESS_JOIN_HUE_LOWER_UPPER_RED+TYPE_SAVED, join_lower_upper_red_hue );

    /*
    Mat kernel = (Mat_<uchar>(5,5) << 0,1,0,1,1,1,0,1,0);
    erode(join_lower_upper_red_hue, join_lower_upper_red_hue, kernel);
    imshow("erode", join_lower_upper_red_hue);

    kernel = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
    dilate(join_lower_upper_red_hue, join_lower_upper_red_hue, kernel);
    imshow(TITLE_ANGKA_MERAH_PREPOCESS_DILATE, join_lower_upper_red_hue);
    imwrite( LOCATION_SAVED+TITLE_ANGKA_MERAH_PREPOCESS_DILATE+TYPE_SAVED, join_lower_upper_red_hue );
    */

    return join_lower_upper_red_hue;
}
/*
void angkaExtraction_preprocess2() {

    /// blur untuk menghilangkan noise
    if (value_angkaExtraction_preprocess_blurKernel%2==0) {
        return;
    }
    blur( img, img, Size(value_angkaExtraction_preprocess_blurKernel,value_angkaExtraction_preprocess_blurKernel) );
    imshow( "angka - Blur", img );

    /// Segmentation - Sobel
    //check if kernel is not odd
    if (value_angkaExtraction_preprocess_sobelKernel%2==0) {
        return;
    }
    Sobel(img, img, -1, 1, 0, value_angkaExtraction_preprocess_sobelKernel);
    imshow( "angka - Sobel", img );

    /// Morphological - get structuring elemnt
    Mat se = getStructuringElement(MORPH_RECT, Size(value_angkaExtraction_preprocess_morphological_width, value_angkaExtraction_preprocess_morphological_height));
    morphologyEx(img, img, MORPH_CLOSE, se);
    imshow( "Structuring Element", img );

}
*/

vector<Mat> angkaExtraction_featureExtraction(Mat img, string interestWindowName) {
    vector<Mat> listImgExtracted;
    Mat threshold_result;
    threshold( img, threshold_result, value_angkaExtraction_featureExtraction_thresh, max_value_angkaExtraction_featureExtraction_thresh, THRESH_BINARY );
    imshow( interestWindowName+TITLE_ANGKA_FEATURE_EXTRACTION_THRESHOLD, threshold_result );
    imwrite( LOCATION_SAVED+interestWindowName+TITLE_ANGKA_FEATURE_EXTRACTION_THRESHOLD+TYPE_SAVED, threshold_result );

    /// Find contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours( threshold_result, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

    /// Find the rotated rectangles for each contour
    vector<RotatedRect> minRect( contours.size() );
    for( int i = 0; i < contours.size(); i++ )
    {
        minRect[i] = minAreaRect( Mat(contours[i]) );
    }

    ///draw contours and rotated rectangle for visualization
    drawContours(contours, minRect, interestWindowName+TITLE_ANGKA_FEATURE_EXTRACTION_CONTOURS, threshold_result.size());

    /// Get the interest
    vector<RotatedRect> interestImage_rect_src;
    getInterestAngka(value_angkaExtraction_featureExtraction_ratioRectangleLengthWidth/10, minRect, hierarchy, &interestImage_rect_src, threshold_result.size());

    ///draw interest for visualization
    drawInterestAngka(interestImage_rect_src, interestWindowName+TITLE_ANGKA_FEATURE_EXTRACTION_CONTOURS_INTEREST, threshold_result.size());

    /// Create a new image
    /// remap kumpulan interest angka menjadi gambar baru untuk diklasifikasi:
    for(int i=interestImage_rect_src.size()-1; i>=0; i--) {
        Mat imageAngkaExtracted;

        Point2f interestImage_points_src[4];
        interestImage_rect_src[i].points( interestImage_points_src );
        float lengthX, lengthY;
        convertToClockWise(interestImage_rect_src[i].center, interestImage_points_src, interestImage_points_src);
        getRectangleMaxLengthWidth(interestImage_points_src, &lengthX, &lengthY);

        Point2f interestImage_points_dst[4];
        getDstPoint(interestImage_points_dst, Size(lengthX,lengthY));

        warpPerspective(threshold_result, imageAngkaExtracted, getPerspectiveTransform(interestImage_points_src, interestImage_points_dst), Size(lengthX, lengthY), INTER_LINEAR, BORDER_CONSTANT, CV_RGB(255,255,255));
        //namedWindow( "Feature Extraction - Extracted Image", CV_WINDOW_AUTOSIZE );
        //imshow( "Feature Extraction - Extracted Image", imageBlokAngkaExtracted );
        //imwrite( LOCATION_METERAN_AIR_ANGKA_EXTRACTED+to_string(i)+".jpg", imageAngkaExtracted );

        listImgExtracted.push_back(imageAngkaExtracted);
    }

    return listImgExtracted;
}


/// get the interest object (angka meteran air)
void getInterestAngka(double ratioXY, vector<RotatedRect> rect, vector<Vec4i> hierarchy, vector<RotatedRect> *interestImage_point_src, Size sizeBlokAngka) {
    //cout << "getInterestAngka " << endl;

    int areaBlokAngka = sizeBlokAngka.width * sizeBlokAngka.height;
    //cout << "area blok angka = " << areaBlokAngka <<endl;
    float defaultAreaAngka = areaBlokAngka/35;
    float defaultAreaAngkaRangeMin = defaultAreaAngka - (0.6*defaultAreaAngka);
    float defaultAreaAngkaRangeMax = defaultAreaAngka + (0.6*defaultAreaAngka);
    //cout <<  "default area angka = " << defaultAreaAngka << endl;
    //cout << "range angka = " << defaultAreaAngkaRangeMin << " " << defaultAreaAngkaRangeMax << endl;

    for( int i = 0; i< rect.size(); i++ )
    {
        //convert rectangle to point
        Point2f rect_points[4];
        rect[i].points( rect_points );

        //get rectangle max length
        float maxLengthX, maxLengthY;
        convertToClockWise(rect[i].center, rect_points, rect_points);
        getRectangleMaxLengthWidth(rect_points, &maxLengthX, &maxLengthY);

        //get area
        float area = maxLengthX*maxLengthY;

        double minRatioXY = ratioXY - (0.3*ratioXY);
        double maxRatioXY = ratioXY + (0.1*ratioXY);

        //jika dia adalah angka (interest) yaitu bentuk persegi panjang, dan punya luas yang didalam ketetapan range luas
        if ((maxLengthY>(maxLengthX*minRatioXY) && maxLengthY<(maxLengthX*maxRatioXY)) && area>=defaultAreaAngkaRangeMin && area<=defaultAreaAngkaRangeMax) {
            //cout << endl << "area= "<<area << " " << endl;
            //cek apakah dia adalah child dari contour lain
            // Misal angka 0. Contour mendeteksi angka 0 menjadi lingkar luar angka 0, dan lingkar dalam (lubang) angka 0
            // Child angka 0 (lubangnya) tidak akan menjadi interest
            // oleh sebab itu dicek, apakah dia punya parent (jika ya kemungkinan dia adalah lubang angka 0)
            // jika punya, cek lagi apakah parentnya itu adalah benar contour si lingkar luar angka 0
            if(hierarchy[i][3]!=-1) {
                //cout << "hierarcy contour index " << i << " = " << hierarchy[i][3] << endl;
                Point2f rect_parent_points[4];
                rect[hierarchy[i][3]].points( rect_parent_points );
                //cout << "rect_parent_points= " << rect_parent_points[i] << endl;

                //hitung luas parent
                float maxParentLengthX, maxParentLengthY;
                convertToClockWise(rect[i].center, rect_parent_points, rect_parent_points);
                getRectangleMaxLengthWidth(rect_parent_points, &maxParentLengthX, &maxParentLengthY);
                float areaParent = maxParentLengthX*maxParentLengthY;
                //cout << "parent area = " << areaParent << endl;

                //bisa jadi parent adalah border blok angka meteran air
                //oleh sebab itu cek jika parent adalah angka, yaitu punya luas yang didalam ketetapan range luas
                if(areaParent>=defaultAreaAngkaRangeMin && areaParent<=defaultAreaAngkaRangeMax) {
                    //ya dia punya parent yaitu angka
                    //maka dia tidak termasuk interest
                    //(*interestImage_point_src).push_back(rect[i]);
                } else {
                    (*interestImage_point_src).push_back(rect[i]);
                }

            }
            //jika tidak punya parent, maka dia benar angka seutuhnya.
            else {
                (*interestImage_point_src).push_back(rect[i]);
            }
        }

    }
}

void drawInterestAngka(vector<RotatedRect> rect, String windowName, Size windowSize) {
    Mat window = Mat::zeros( windowSize, CV_8UC3 );
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );

    for( int i = 0; i< rect.size(); i++ )
    {
        Point2f rect_points[4];
        rect[i].points( rect_points );

        for( int j = 0; j < 4; j++ ) {
            line( window, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
        }
    }

    /// Show in a window
    namedWindow( windowName, CV_WINDOW_AUTOSIZE );
    imshow( windowName, window );
    imwrite( LOCATION_SAVED+windowName+TYPE_SAVED, window );
}
