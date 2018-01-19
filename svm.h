/*

this program is download from https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/
modified by Albert Alfrianta - albert.brucelee@gmail.com

SPREAD OUT THE POWER OF OPEN SOURCE!

*/

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>

using namespace cv::ml;
using namespace cv;
using namespace std;

int SZ = 20;
float affineFlags = WARP_INVERSE_MAP|INTER_LINEAR;
#define PERCENT_COUNT_IMAGE_TRAIN 1

Mat deskew(Mat& img){
    Moments m = moments(img);
    if(abs(m.mu02) < 1e-2){
        return img.clone();
    }
    float skew = m.mu11/m.mu02;
    Mat warpMat = (Mat_<float>(2,3) << 1, skew, -0.5*SZ*skew, 0, 1, 0);
    Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
    warpAffine(img, imgOut, warpMat, imgOut.size(),affineFlags);
    return imgOut;
}

void loadTrainTestLabel(string pathName, vector<Mat> &trainCells, vector<Mat> &testCells,vector<int> &trainLabels, vector<int> &testLabels)
{

    Mat img = imread(pathName,CV_LOAD_IMAGE_GRAYSCALE);;
    int ImgCount = 0;
    for(int i = 0; i < img.rows; i = i + SZ)
    {
        for(int j = 0; j < img.cols; j = j + SZ)
        {
            Mat digitImg = (img.colRange(j,j+SZ).rowRange(i,i+SZ)).clone();
            if(j < int(PERCENT_COUNT_IMAGE_TRAIN*img.cols))
            {
                trainCells.push_back(digitImg);
            }
            else
            {
                testCells.push_back(digitImg);
            }
            ImgCount++;
        }
    }

    cout << "Image Count for svm train : " << round(PERCENT_COUNT_IMAGE_TRAIN * ImgCount) << endl;
    float digitClassNumber = 0;
    for(int z=0; z<(int)(PERCENT_COUNT_IMAGE_TRAIN*ImgCount); z++){
        if(z % ((int) round(PERCENT_COUNT_IMAGE_TRAIN * ImgCount / 10)) == 0 && z != 0){
            digitClassNumber = digitClassNumber + 1;
            }
        trainLabels.push_back(digitClassNumber);
    }
    digitClassNumber = 0;
    for(int z=0; z<(int)((1-PERCENT_COUNT_IMAGE_TRAIN)*ImgCount); z++){
        if(z % ((int)round((1-PERCENT_COUNT_IMAGE_TRAIN) * ImgCount / 10)) == 0 && z != 0){
            digitClassNumber = digitClassNumber + 1;
            }
        testLabels.push_back(digitClassNumber);
    }
}

void CreateDeskewedTrainTest(vector<Mat> &deskewedTrainCells,vector<Mat> &deskewedTestCells, vector<Mat> &trainCells, vector<Mat> &testCells){


    for(int i=0;i<trainCells.size();i++){

        Mat deskewedImg = deskew(trainCells[i]);
        deskewedTrainCells.push_back(deskewedImg);
    }

    for(int i=0;i<testCells.size();i++){

        Mat deskewedImg = deskew(testCells[i]);
        deskewedTestCells.push_back(deskewedImg);
    }
}

HOGDescriptor hog(
        Size(20,20), //winSize
        Size(8,8), //blocksize
        Size(4,4), //blockStride,
        Size(8,8), //cellSize,
                 9, //nbins,
                  1, //derivAper,
                 -1, //winSigma,
                  0, //histogramNormType,
                0.2, //L2HysThresh,
                  0,//gammal correction,
                  64,//nlevels=64
                  1);
void CreateTrainTestHOG(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, vector<Mat> &deskewedtrainCells, vector<Mat> &deskewedtestCells){

    for(int y=0;y<deskewedtrainCells.size();y++){
        vector<float> descriptors;
        hog.compute(deskewedtrainCells[y],descriptors);
        trainHOG.push_back(descriptors);
    }

    for(int y=0;y<deskewedtestCells.size();y++){

        vector<float> descriptors;
        hog.compute(deskewedtestCells[y],descriptors);
        testHOG.push_back(descriptors);
    }
}

void ConvertVectortoMatrix(vector<vector<float> > &trainHOG, vector<vector<float> > &testHOG, Mat &trainMat, Mat &testMat)
{

    int descriptor_size = trainHOG[0].size();

    for(int i = 0;i<trainHOG.size();i++){
        for(int j = 0;j<descriptor_size;j++){
           trainMat.at<float>(i,j) = trainHOG[i][j];
        }
    }
    for(int i = 0;i<testHOG.size();i++){
        for(int j = 0;j<descriptor_size;j++){
            testMat.at<float>(i,j) = testHOG[i][j];
        }
    }
}

void getSVMParams(SVM *svm)
{
    cout << "Kernel type     : " << svm->getKernelType() << endl;
    cout << "Type            : " << svm->getType() << endl;
    cout << "C               : " << svm->getC() << endl;
    cout << "Degree          : " << svm->getDegree() << endl;
    cout << "Nu              : " << svm->getNu() << endl;
    cout << "Gamma           : " << svm->getGamma() << endl;
}

Ptr<SVM> svmInit(float C, float gamma)
{
  Ptr<SVM> svm = SVM::create();
  svm->setGamma(gamma);
  svm->setC(C);
  svm->setKernel(SVM::RBF);
  svm->setType(SVM::C_SVC);

  return svm;
}

void svmTrain(Ptr<SVM> svm, Mat &trainMat, vector<int> &trainLabels)
{
  Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
  svm->train(td);
  svm->save(LOCATION_SAVED_MODEL_SVM);
}

void svmPredict(Ptr<SVM> svm, Mat &testResponse, Mat &testMat )
{
  svm->predict(testMat, testResponse);
}

void SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels)
{
  for(int i = 0; i < testResponse.rows; i++)
  {
    // cout << testResponse.at<float>(i,0) << " " << testLabels[i] << endl;
    if(testResponse.at<float>(i,0) == testLabels[i]) {
      //cout << testResponse.at<float>(i,0) << " ";
      count = count + 1;
    }
  }
  accuracy = (count/testResponse.rows)*100;
}


Ptr<SVM> svmClassification(string pathTrainImage)
{

    vector<Mat> trainCells;
    vector<Mat> testCells;
    vector<int> trainLabels;
    vector<int> testLabels;
    loadTrainTestLabel(pathTrainImage,trainCells,testCells,trainLabels,testLabels);

    vector<Mat> deskewedTrainCells;
    vector<Mat> deskewedTestCells;
    CreateDeskewedTrainTest(deskewedTrainCells,deskewedTestCells,trainCells,testCells);

    std::vector<std::vector<float> > trainHOG;
    std::vector<std::vector<float> > testHOG;
    CreateTrainTestHOG(trainHOG,testHOG,deskewedTrainCells,deskewedTestCells);

    int descriptor_size = trainHOG[0].size();
    //cout << "Descriptor Size : " << trainHOG[0].size() << endl;

    Mat trainMat(trainHOG.size(),descriptor_size,CV_32FC1);
    Mat testMat(testHOG.size(),descriptor_size,CV_32FC1);

    ConvertVectortoMatrix(trainHOG,testHOG,trainMat,testMat);

    float C = 12.5, gamma = 0.5;

    Mat testResponse;
    Ptr<SVM> model = svmInit(C, gamma);

    ///////////  SVM Training  ////////////////
    svmTrain(model, trainMat, trainLabels);


    /*
    ///////////  SVM Testing  ////////////////
    svmPredict(model, testResponse, testMat);

    ////////////// Find Accuracy   ///////////
    float count = 0;
    float accuracy = 0 ;
    getSVMParams(model);
    SVMevaluate(testResponse, count, accuracy, testLabels);

    cout << "the accuracy is :" << accuracy << endl;
    */

    return model;
}


float testClassify(Ptr<SVM> model, Mat testDigit)
{
    ///////////  SVM Testing  ////////////////

    //loadTrainTestLabel

    resize(testDigit, testDigit, Size(SZ,SZ));
    //CreateDeskewedTrainTest
    Mat testDigitDeskew = deskew(testDigit);
    //CreateTrainTestHOG
    vector<float> testDigitHOG;
    hog.compute(testDigitDeskew,testDigitHOG);
    //Create testDigitMat
    //cout << "Descriptor Size : " << testDigitHOG.size() << endl;
    Mat testDigitMat(1,testDigitHOG.size(),CV_32FC1);
    //ConvertVectortoMatrix
    for(int j = 0;j<testDigitHOG.size();j++){
       testDigitMat.at<float>(j) = testDigitHOG[j];
    }
    //predict
    Mat testResponse;
    svmPredict(model, testResponse, testDigitMat);
    //cout << "#Result= " << testResponse.at<float>(0,0) << " " << (i) << endl;
    return testResponse.at<float>(0,0);
}


