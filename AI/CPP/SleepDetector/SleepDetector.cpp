// OpenCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

// Dlib
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/opencv/to_open_cv.h>
#include <dlib/opencv.h>
#include <dlib/serialize.h>

// std
#include <iostream>
#include <fstream>
#include <malloc.h>
#include <List>

using namespace std;
using namespace cv;
using namespace dlib;

float GetPointDistance(long points[][2], int p1, int p2);
float GetTiltY(long points[][2], int p1, int p2);

int OnFrame(int32_t width, int32_t height, uint8_t* bytes, dlib::frontal_face_detector* detector, dlib::shape_predictor* predictor);

int OnFrameC(Mat frame, dlib::frontal_face_detector* detector, dlib::shape_predictor* predictor);

frontal_face_detector*	MakeDetector();
shape_predictor*		MakePredictor();

void DestroyDetector(frontal_face_detector& detector);
void DestroyPredictor(shape_predictor& predictor);

#define thresh 5.3*5.3
#define tiltThresh 25

// MAIN //////////////

int main()
{
	Mat frame;
	VideoCapture cap(0, cv::CAP_ANY);
	frontal_face_detector* detector = MakeDetector();
	shape_predictor* predictor = MakePredictor();
	//frontal_face_detector detector = get_frontal_face_detector();
	//shape_predictor predictor;
	//dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> predictor;

	while (cap.isOpened())
	{
		cap.read(frame);
		int blinked = OnFrameC(frame, detector, predictor);

		cout << blinked << endl;

		cv::imshow("Frame", frame);

		if (waitKey(1) >= 0)
			break;
	}

	DestroyDetector(*detector);
	DestroyPredictor(*predictor);
}

//////////////////////

float GetPointDistance(long points[][2], int p1, int p2)
{ return (points[p1][0] - points[p2][0]) * (points[p1][0] - points[p2][0]) + (points[p1][1] - points[p2][1]) * (points[p1][1] - points[p2][1]); }

float GetTiltY(long points[][2], int p1, int p2)
{ return std::abs(points[p1][1] - points[p2][1]); }

int OnFrame(int32_t width, int32_t height, uint8_t* bytes, dlib::frontal_face_detector* detector, shape_predictor* predictor)
{
	Mat gray;
	Mat frame = Mat::Mat(width, height, CV_8U, &bytes).clone();
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	cv_image<unsigned char> d_gray(gray);
	std::vector<dlib::rectangle> faces = (*detector)(d_gray);

	if (faces.size() <= 0)
		return -1;

	long points[68][2];
	auto face = faces[0];
	auto landmarks = (*predictor)(d_gray, face);
	for (int i = 0; i < 68; i++)
	{
		points[i][0] = landmarks.part(i).x();
		points[i][1] = landmarks.part(i).y();
	}

	auto eyeWidth = GetPointDistance(points, 36, 39);
	auto eyeHeight = GetPointDistance(points, 37, 41);

	if (eyeWidth / eyeHeight > thresh)
		return 0;
	else if (GetTiltY(points, 39, 42))
		return 0;
	else
		return 1;
}

int OnFrameC(Mat frame, dlib::frontal_face_detector* detector, dlib::shape_predictor* predictor)
{
	Mat gray;
	cvtColor(frame, gray, COLOR_BGR2GRAY);
	cv_image<unsigned char> d_gray(gray);
	std::vector<dlib::rectangle> faces = (*detector)(d_gray);

	if (faces.size() <= 0)
		return -1;

	long points[68][2];
	auto face = faces[0];
	auto landmarks = (*predictor)(d_gray, face);
	for (int i = 0; i < 68; i++)
	{
		points[i][0] = landmarks.part(i).x();
		points[i][1] = landmarks.part(i).y();

		cv::circle(frame, Point(points[i][0], points[i][1]), 3, Scalar(0, 255, 0), cv::FILLED);
	}

	auto eyeWidth = GetPointDistance(points, 36, 39);
	auto eyeHeight = GetPointDistance(points, 37, 41);

	if (eyeWidth / eyeHeight > thresh)
		return 0;
	else if (GetTiltY(points, 39, 42))
		return 0;
	else
		return 1;
}

frontal_face_detector* MakeDetector() { frontal_face_detector* detector = new frontal_face_detector(get_frontal_face_detector()); return detector; }

shape_predictor* MakePredictor()
{
	shape_predictor* predictor = new shape_predictor();
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> (*predictor);
	return predictor;
}

void DestroyDetector(frontal_face_detector& detector)
{
	delete &detector;
}

void DestroyPredictor(shape_predictor& predictor)
{
	delete &predictor;
}