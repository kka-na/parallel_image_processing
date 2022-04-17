#ifndef Detection_H
#define Detection_H

#include <omp.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;
using namespace dnn::dnn4_v20201117;

class Detection
{
private:
    float confThreshold;
    float nmsThreshold;
    int inpWidth;
    int inpHeight;
    vector<string> classes;

public:
    int FaceDetection(Mat _frame, string windowName)
    {
        CascadeClassifier face_cascade("../xml/face.xml");
        CascadeClassifier eyes_cascade("../xml/eyes.xml");
        double fstart = omp_get_wtime();
        while (true)
        {

            Mat frame = _frame.clone();
            resize(frame, frame, Size(480, 480));
            Mat frame_gray;
            cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
            equalizeHist(frame_gray, frame_gray);
            vector<Rect> faces;
            face_cascade.detectMultiScale(frame_gray, faces);
            int i, j;
            for (i = 0; i < faces.size(); i++)
            {
                Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
                ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
                Mat faceROI = frame_gray(faces[i]);
                vector<Rect> eyes;
                eyes_cascade.detectMultiScale(faceROI, eyes);
                for (j = 0; j < eyes.size(); j++)
                {
                    Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
                    int radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
                    circle(frame, eye_center, radius, Scalar(255, 0, 0), 4);
                }
            }

            double fend = omp_get_wtime();
            double fps = (0.1 / (fend - fstart));
            putText(frame, to_string(fps) + "FPS", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            imshow(windowName, frame);
            if (waitKey(33) == 27)
                break;
            fstart = fend;
        }
        return 0;
    }
    int HumanDetection(Mat _frame, string windowName)
    {
        confThreshold = 0.5;
        nmsThreshold = 0.4;
        inpWidth = 480;
        inpHeight = 480;

        // Load names of classes
        string classesFile = "../dnn/coco.names";
        ifstream ifs(classesFile.c_str());
        string line;
        while (getline(ifs, line))
            classes.push_back(line);

        // Give the configuration and weight files for the model
        String modelConfiguration = "../dnn/yolov3.cfg";
        String modelWeights = "../dnn/yolov3.weights";

        // Load the network
        Net net = readNetFromDarknet(modelConfiguration, modelWeights);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);

        Mat blob;

        double fstart = omp_get_wtime();
        while (true)
        {
            Mat frame = _frame.clone();
            resize(frame, frame, Size(inpWidth, inpHeight));

            blobFromImage(frame, blob, 1 / 255.0, cv::Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
            net.setInput(blob);
            vector<Mat> outs;
            net.forward(outs, getOutputsNames(net));

            postprocess(frame, outs);
            Mat detectedFrame;
            frame.convertTo(detectedFrame, CV_8U);

            double fend = omp_get_wtime();
            double fps = (0.1 / (fend - fstart));
            putText(frame, to_string(fps) + "FPS", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            imshow(windowName, frame);
            if (waitKey(3) == 27)
                break;
            fstart = fend;
        }
        return 0;
    }
    // Get the names of the output layers
    vector<String> getOutputsNames(const Net &net)
    {
        static vector<String> names;
        if (names.empty())
        {
            vector<int> outLayers = net.getUnconnectedOutLayers();
            vector<String> layersNames = net.getLayerNames();
            names.resize(outLayers.size());
            for (size_t i = 0; i < outLayers.size(); ++i)
                names[i] = layersNames[outLayers[i] - 1];
        }
        return names;
    }
    void postprocess(Mat &frame, const vector<Mat> &outs)
    {
        vector<int> classIds;
        vector<float> confidences;
        vector<Rect> boxes;

        for (size_t i = 0; i < outs.size(); ++i)
        {
            float *data = (float *)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                // Get the value and location of the maximum score
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
                if (confidence > confThreshold)
                {
                    int centerX = (int)(data[0] * frame.cols);
                    int centerY = (int)(data[1] * frame.rows);
                    int width = (int)(data[2] * frame.cols);
                    int height = (int)(data[3] * frame.rows);
                    int left = centerX - width / 2;
                    int top = centerY - height / 2;

                    classIds.push_back(classIdPoint.x);
                    confidences.push_back((float)confidence);
                    boxes.push_back(Rect(left, top, width, height));
                }
            }
        }

        vector<int> indices;
        NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
        for (size_t i = 0; i < indices.size(); ++i)
        {
            int idx = indices[i];
            Rect box = boxes[idx];
            if (classIds[idx] == 0)
            {
                drawPred(classIds[idx], confidences[idx], box.x, box.y,
                         box.x + box.width, box.y + box.height, frame);
            }
        }
    }
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat &frame)
    {
        rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

        string label = format("%.2f", conf);
        if (!classes.empty())
        {
            CV_Assert(classId < (int)classes.size());
            label = classes[classId] + ":" + label;
        }

        // Display the label at the top of the bounding box
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        top = max(top, labelSize.height);
        rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
        putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
    }
};

#endif
