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


const string TITLE_BLOK_ANGKA_PREPOCESS_TO_GRAY = "BlokAngka-Preprocess-to_gray";
const string TITLE_BLOK_ANGKA_PREPOCESS_BLUR = "BlokAngka-Preprocess-blur";
const string TITLE_BLOK_ANGKA_PREPOCESS_SEGMENTATION = "BlokAngka-Preprocess-segmentation_sobel";
const string TITLE_BLOK_ANGKA_PREPOCESS_MORPHOLOGICAL = "BlokAngka-Preprocess-morphological";
const string TITLE_BLOK_ANGKA_FEATURE_EXTRACTION_THRESHOLD = "BlokAngka-FeatureExtraction-threshold";
const string TITLE_BLOK_ANGKA_FEATURE_EXTRACTION_CONTOURS = "BlokAngka-FeatureExtraction-contours";
const string TITLE_BLOK_ANGKA_FEATURE_EXTRACTION_CONTOURS_INTEREST = "BlokAngka-FeatureExtraction-contours_interest";
const string TITLE_BLOK_ANGKA_FEATURE_EXTRACTION_RESULT = "BlokAngka-FeatureExtraction-result";

const string TRACKBAR_WINDOW = "Trackbar BlokAngkaExtract";

Mat src_blokAngkaExtraction;

int value_blokAngkaExtraction_preprocess_blurKernel = 5;
int max_value_blokAngkaExtraction_preprocess_blurKernel = 10;

int value_blokAngkaExtraction_preprocess_sobelKernel = 3;
int max_value_blokAngkaExtraction_preprocess_sobelKernel = 31;

int value_blokAngkaExtraction_preprocess_morphological_width = 16;
int max_value_blokAngkaExtraction_preprocess_morphological_width = 20;

int value_blokAngkaExtraction_preprocess_morphological_height = 4;
int max_value_blokAngkaExtraction_preprocess_morphological_height = 20;

int value_blokAngkaExtraction_featureExtraction_thresh = 46;
int max_value_blokAngkaExtraction_featureExtraction_thresh = 255;

int value_blokAngkaExtraction_featureExtraction_ratioRectangleLengthWidth = 3;
int max_value_blokAngkaExtraction_featureExtraction_ratioRectangleLengthWidth = 10;


void blokAngkaExtractionInit();
void blokAngkaExtraction(int, void*);
void blokAngkaExtraction_preprocess();
void blokAngkaExtraction_featureExtraction();
void drawContours(vector<vector<Point>>contours, vector<RotatedRect> rect, String windowName, Size windowSize);
void getInterest(double ratioXY, vector<vector<Point>>contours, vector<RotatedRect> rect, int *returnContourIndexInterest, Point2f *rect_points_src, Point2f *rect_points_dst, int *returnMaxLengthX, int *returnMaxLengthY);
void convertToClockWise(Point2f center_interest, Point2f *point_interest, Point2f *return_point_interest);
void getRectangleMaxLengthWidth(Point2f *point_interest, float *returnMaxLengthX, float *returnMaxLengthY);
void getDstPoint(Point2f *interestImage_point_dst, Size sizeInterestImage);
void drawInterest(vector<vector<Point>>contours, int contourIndex, Point2f *rect_points_src, String windowName, Size windowSize);

void blokAngkaExtractionInit() {

    namedWindow( TRACKBAR_WINDOW, CV_WINDOW_AUTOSIZE );

    createTrackbar( " Blur:", TRACKBAR_WINDOW, &value_blokAngkaExtraction_preprocess_blurKernel, max_value_blokAngkaExtraction_preprocess_blurKernel, blokAngkaExtraction );
    createTrackbar( " Sobel:", TRACKBAR_WINDOW, &value_blokAngkaExtraction_preprocess_sobelKernel, max_value_blokAngkaExtraction_preprocess_sobelKernel, blokAngkaExtraction );
    createTrackbar( " Morphological Width:", TRACKBAR_WINDOW, &value_blokAngkaExtraction_preprocess_morphological_width, max_value_blokAngkaExtraction_preprocess_morphological_width, blokAngkaExtraction );
    createTrackbar( " Morphological Height:", TRACKBAR_WINDOW, &value_blokAngkaExtraction_preprocess_morphological_height, max_value_blokAngkaExtraction_preprocess_morphological_width, blokAngkaExtraction );
    createTrackbar( " Threshold:", TRACKBAR_WINDOW, &value_blokAngkaExtraction_featureExtraction_thresh, max_value_blokAngkaExtraction_featureExtraction_thresh, blokAngkaExtraction );
    createTrackbar( " Ratio Rectangle Length Width:", TRACKBAR_WINDOW, &value_blokAngkaExtraction_featureExtraction_ratioRectangleLengthWidth, max_value_blokAngkaExtraction_featureExtraction_ratioRectangleLengthWidth, blokAngkaExtraction );
}

void blokAngkaExtraction(int, void*) {
    blokAngkaExtraction_preprocess();
    blokAngkaExtraction_featureExtraction();
}

void blokAngkaExtraction_preprocess() {
    /// Convert image to gray
    cvtColor( src, src_blokAngkaExtraction, CV_BGR2GRAY );
    imshow( TITLE_BLOK_ANGKA_PREPOCESS_TO_GRAY, src_blokAngkaExtraction );
    imwrite( LOCATION_SAVED+TITLE_BLOK_ANGKA_PREPOCESS_TO_GRAY+TYPE_SAVED, src_blokAngkaExtraction );

    /// blur untuk menghilangkan noise
    if (value_blokAngkaExtraction_preprocess_blurKernel%2==0) {
        return;
    }
    blur( src_blokAngkaExtraction, src_blokAngkaExtraction, Size(value_blokAngkaExtraction_preprocess_blurKernel,value_blokAngkaExtraction_preprocess_blurKernel) );
    imshow( TITLE_BLOK_ANGKA_PREPOCESS_BLUR, src_blokAngkaExtraction );
    imwrite( LOCATION_SAVED+TITLE_BLOK_ANGKA_PREPOCESS_BLUR+TYPE_SAVED, src_blokAngkaExtraction );

    /// Segmentation - Sobel
    /// Threshold the resultant image using strict threshold or OTSU's binarization.
    //check if kernel is not odd
    if (value_blokAngkaExtraction_preprocess_sobelKernel%2==0) {
        return;
    }
    Sobel(src_blokAngkaExtraction, src_blokAngkaExtraction, -1, 1, 0, value_blokAngkaExtraction_preprocess_sobelKernel);
    imshow( TITLE_BLOK_ANGKA_PREPOCESS_SEGMENTATION, src_blokAngkaExtraction );
    imwrite( LOCATION_SAVED+TITLE_BLOK_ANGKA_PREPOCESS_SEGMENTATION+TYPE_SAVED, src_blokAngkaExtraction );

    /// Morphological Closing operation using suitable structuring element.
    Mat se = getStructuringElement(MORPH_RECT, Size(value_blokAngkaExtraction_preprocess_morphological_width, value_blokAngkaExtraction_preprocess_morphological_height));
    morphologyEx(src_blokAngkaExtraction, src_blokAngkaExtraction, MORPH_CLOSE, se);
    imshow( TITLE_BLOK_ANGKA_PREPOCESS_MORPHOLOGICAL, src_blokAngkaExtraction );
    imwrite( LOCATION_SAVED+TITLE_BLOK_ANGKA_PREPOCESS_MORPHOLOGICAL+TYPE_SAVED, src_blokAngkaExtraction );
}

void blokAngkaExtraction_featureExtraction() {
    /// Detect edges using Threshold
    Mat threshold_result;
    threshold( src_blokAngkaExtraction, threshold_result, value_blokAngkaExtraction_featureExtraction_thresh, max_value_blokAngkaExtraction_featureExtraction_thresh, THRESH_BINARY );
    imshow( TITLE_BLOK_ANGKA_FEATURE_EXTRACTION_THRESHOLD, threshold_result );
    imwrite( LOCATION_SAVED+TITLE_BLOK_ANGKA_FEATURE_EXTRACTION_THRESHOLD+TYPE_SAVED, threshold_result );

    /// Find external contours
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
    drawContours(contours, minRect, TITLE_BLOK_ANGKA_FEATURE_EXTRACTION_CONTOURS, threshold_result.size());

    /// Get the interest
    int maxLengthX, maxLengthY, contourIndexInterest;
    Point2f interestImage_points_src[4], interestImage_points_dst[4];
    getInterest(value_blokAngkaExtraction_featureExtraction_ratioRectangleLengthWidth, contours, minRect, &contourIndexInterest, interestImage_points_src, interestImage_points_dst, &maxLengthX, &maxLengthY);

    ///draw interest for visualization
    drawInterest(contours, contourIndexInterest, interestImage_points_src, TITLE_BLOK_ANGKA_FEATURE_EXTRACTION_CONTOURS_INTEREST, threshold_result.size());

    /// Create a new image
    /// whe have the corner point topLeft,topRight,bottomLeft,bottomRight
    /// so we remap the original image with that corner and also make the image to be top view perspective:
    warpPerspective(src, imageBlokAngkaExtracted, getPerspectiveTransform(interestImage_points_src, interestImage_points_dst), Size(maxLengthX, maxLengthY), INTER_LINEAR, BORDER_CONSTANT, CV_RGB(255,255,255));
    imshow( TITLE_BLOK_ANGKA_FEATURE_EXTRACTION_RESULT, imageBlokAngkaExtracted );
    imwrite( LOCATION_SAVED+TITLE_BLOK_ANGKA_FEATURE_EXTRACTION_RESULT+TYPE_SAVED, threshold_result );

}


///draw contours only for visualisation
void drawContours(vector<vector<Point>>contours, vector<RotatedRect> rect, String windowName, Size windowSize) {
    Mat window = Mat::zeros( windowSize, CV_8UC3 );

    for( int i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        // contour
        drawContours( window, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point() );

        // rotated rectangle
        Point2f rect_points[4]; rect[i].points( rect_points );
        for( int j = 0; j < 4; j++ ) {
            //cout << rect_points[j] <<endl;
            line( window, rect_points[j], rect_points[(j+1)%4], color, 1, 8 );
        }
    }

    /// Show in a window
    namedWindow( windowName, CV_WINDOW_AUTOSIZE );
    imshow( windowName, window );
    imwrite( LOCATION_SAVED+windowName+TYPE_SAVED, window );
}


/// get the interest object (angka meteran air)
void getInterest(double ratioXY, vector<vector<Point>>contours, vector<RotatedRect> rect, int *returnContourIndexInterest, Point2f *interestImage_point_src, Point2f *interestImage_point_dst, int *returnMaxLengthX, int *returnMaxLengthY) {

    int contourIndexInterest;
    double largestArea = 0;

    /// Get the longest rectangle
    for( int i = 0; i< contours.size(); i++ )
    {
        //get rectangle area
        double area = contourArea(contours[i],false);

        //convert rectangle to point
        Point2f rect_points[4];
        rect[i].points( rect_points );

        //get rectangle max length
        float maxLengthX, maxLengthY;
        convertToClockWise(rect[i].center, rect_points, rect_points);
        getRectangleMaxLengthWidth(rect_points, &maxLengthX, &maxLengthY);

        // if it is rectangle (length > 3 * width)
        // and it is the biggest rectangle (angka meteran air adalah rectangle terbesar)
        if( (maxLengthX>maxLengthY*ratioXY || maxLengthY>maxLengthX*ratioXY) && area>largestArea) {

            contourIndexInterest = i;
            largestArea = area;

            ///get the contour point of interest image
            rect[i].points(interestImage_point_src);
            convertToClockWise(rect[i].center, interestImage_point_src, interestImage_point_src);

            ///get max length and width interest image
            *returnMaxLengthX = maxLengthX;
            *returnMaxLengthY = maxLengthY;

        }
    }

    *returnContourIndexInterest = contourIndexInterest;

    ///get src and dst point of interest image
    getDstPoint(interestImage_point_dst, Size(*returnMaxLengthX, *returnMaxLengthY));
}

void convertToClockWise(Point2f center_interest, Point2f *point_interest, Point2f *return_point_interest) {
    //cout << "Clockwise " << center_interest << endl;
    Point2f ptTopLeft=center_interest, ptBottomLeft=center_interest, ptBottomRight=center_interest, ptTopRight=center_interest;
    for(int i=0; i<4; i++) {
        //cout << point_interest[i] << endl;
        if((ptTopLeft.x==center_interest.x && ptTopLeft.y==center_interest.y)                   && point_interest[i].x<=center_interest.x   && point_interest[i].y<ptBottomLeft.y) {
            ptTopLeft = point_interest[i];
        } else if ((ptTopRight.x==center_interest.x && ptTopRight.y==center_interest.y)         && point_interest[i].x>center_interest.x    && point_interest[i].y<ptBottomRight.y) {
            ptTopRight = point_interest[i];
        } else if((ptBottomRight.x==center_interest.x && ptBottomRight.y==center_interest.y)    && point_interest[i].x>=center_interest.x   && point_interest[i].y>ptTopRight.y) {
            ptBottomRight = point_interest[i];
        } else if((ptBottomLeft.x==center_interest.x && ptBottomLeft.y==center_interest.y)      && point_interest[i].x<center_interest.x    && point_interest[i].y>ptTopLeft.y) {
            ptBottomLeft = point_interest[i];
        }
    }

    return_point_interest[0] = ptTopLeft;
    return_point_interest[1] = ptTopRight;
    return_point_interest[2] = ptBottomRight;
    return_point_interest[3] = ptBottomLeft;
    /*
    cout << "res" << endl;
    cout << return_point_interest[0] << endl;
    cout << return_point_interest[1] << endl;
    cout << return_point_interest[2] << endl;
    cout << return_point_interest[3] << endl;
    cout << "end" << endl;
    */
}

///
void getRectangleMaxLengthWidth(Point2f *point_rects, float *returnMaxLengthX, float *returnMaxLengthY) {
    Point2f ptBottomLeft, ptBottomRight, ptTopLeft, ptTopRight;
    ptTopLeft = point_rects[0];
    ptTopRight = point_rects[1];
    ptBottomRight = point_rects[2];
    ptBottomLeft = point_rects[3];

    /// Now we have the points. Now we can correct the skewed perspective. First, we find the longest edge of the puzzle. The new image will be a square of the length of the longest edge.
    /// Simple code. We calculate the length of each edge. Whenever we find a longer edge, we store its length squared. And finally when we have the longest edge, we do a square root to get its exact length.
    float maxLengthX = norm(ptBottomRight-ptBottomLeft); //(ptBottomRight.x-ptBottomLeft.x);
    float temp = norm(ptTopRight-ptTopLeft); //(ptTopRight.x-ptTopLeft.x);
    if(temp>maxLengthX) maxLengthX = temp;

    float maxLengthY = norm(ptBottomRight-ptTopRight); //(ptBottomRight.y-ptTopRight.y);
    temp = norm(ptBottomLeft-ptTopLeft); //(ptBottomLeft.y-ptTopLeft.y);
    if(temp>maxLengthY) maxLengthY = temp;

    *returnMaxLengthX = maxLengthX;
    *returnMaxLengthY = maxLengthY;
}

void getDstPoint(Point2f *interestImage_point_dst, Size sizeInterestImage) {
    interestImage_point_dst[0] = Point2f(0,0);
    interestImage_point_dst[1] = Point2f(sizeInterestImage.width-1, 0);
    interestImage_point_dst[2] = Point2f(sizeInterestImage.width-1, sizeInterestImage.height-1);
    interestImage_point_dst[3] = Point2f(0, sizeInterestImage.height-1);
}

void drawInterest(vector<vector<Point>>contours, int contourIndex, Point2f *interestImage_point, String windowName, Size windowSize) {
    Mat window = Mat::zeros( windowSize, CV_8UC3 );
    Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );

    drawContours( window, contours, contourIndex, color, 1, 8, vector<Vec4i>(), 0, Point() );

    for( int j = 0; j < 4; j++ ) {
        //cout << interestImage_point[j] << endl;
        line( window, interestImage_point[j], interestImage_point[(j+1)%4], color, 1, 8 );
    }

    /// Show in a window
    namedWindow( windowName, CV_WINDOW_AUTOSIZE );
    imshow( windowName, window );
    imwrite( LOCATION_SAVED+windowName+TYPE_SAVED, window );
}

