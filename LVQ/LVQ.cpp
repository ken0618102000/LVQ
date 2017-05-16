#include<omp.h>
#include<iostream>
#include <iomanip>
#include <time.h>
#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <fstream>
#include "iomanip"

using namespace std;
using namespace cv;

void Draw_data(Mat &i_show_result, Mat i_training_data, CvPoint2D32f i_LVQ_weight[4], int i_Epoch, float i_learning_rate);

int main()
{

	int close_index, draw_index, Epoch = 0;
	float close_distant;
	float learning_rate = 0.1;
	Mat show_result = Mat::zeros(1000, 1000, CV_8UC3);
	Mat training_data = (Mat_<int>(9, 9) <<
		0, 0, 0, 0, 0, 1, 1, 1, 1,
		0, 0, 0, 0, 0, 1, 1, 1, 1,
		2, 2, 2, 2, 2, 2, 2, 1, 1,
		2, 2, 2, 2, 2, 2, 2, 1, 1,
		2, 2, 2, 2, 2, 2, 2, 3, 3,
		2, 2, 2, 2, 2, 2, 2, 3, 3,
		2, 2, 2, 2, 2, 2, 2, 3, 3,
		2, 2, 2, 2, 2, 2, 2, 3, 3,
		2, 2, 2, 2, 2, 2, 2, 3, 3
		);

	CvPoint2D32f LVQ_weight[4];
	LVQ_weight[0].x = 0.2;
	LVQ_weight[0].y = 0.2;
	LVQ_weight[3].x = 0.8;
	LVQ_weight[3].y = 0.8;
	LVQ_weight[1].x = 0.8;
	LVQ_weight[1].y = 0.2;
	LVQ_weight[2].x = 0;
	LVQ_weight[2].y = 0.8;

	Draw_data(show_result, training_data, LVQ_weight, Epoch, learning_rate);

	//training_data.at<float>(i, j)
	while (1)
	{
		for (int i = 0; i < 9; i++)
		{
			for (int j = 0; j < 9; j++)
			{

				close_distant = 100000.0;
				for (int k = 0; k < 4; k++)
				{
					float distant = sqrt(pow(j + 1 - LVQ_weight[k].x * 10, 2) + pow(i + 1 - LVQ_weight[k].y * 10, 2));
					if (close_distant > distant)
					{
						close_index = k;
						close_distant = distant;
					}
				}

				if (close_index == training_data.at<int>(i, j))
				{
					LVQ_weight[close_index].x = LVQ_weight[close_index].x + learning_rate*((float)j / 10 + 0.1 - LVQ_weight[close_index].x);
					LVQ_weight[close_index].y = LVQ_weight[close_index].y + learning_rate*((float)i / 10 + 0.1 - LVQ_weight[close_index].y);
				}
				else
				{
					LVQ_weight[close_index].x = LVQ_weight[close_index].x - learning_rate*((float)j / 10 + 0.1 - LVQ_weight[close_index].x);
					LVQ_weight[close_index].y = LVQ_weight[close_index].y - learning_rate*((float)i / 10 + 0.1 - LVQ_weight[close_index].y);
				}
				//Draw_data(show_result, training_data, LVQ_weight);
			}
		}

		for (int i = 0; i < 9; i++)
		{
			for (int j = 0; j < 9; j++)
			{

				close_distant = 100000.0;
				for (int k = 0; k < 4; k++)
				{
					float distant = sqrt(pow(j + 1 - LVQ_weight[k].x * 10, 2) + pow(i + 1 - LVQ_weight[k].y * 10, 2));
					if (close_distant > distant)
					{
						close_index = k;
						close_distant = distant;
					}
				}

				training_data.at<int>(i, j) = close_index;
			}
		}
		learning_rate = learning_rate / 1.1;
		Epoch++;

		Draw_data(show_result, training_data, LVQ_weight, Epoch, learning_rate);

	}
	system("pause");
	return 0;

}

void Draw_data(Mat &i_show_result, Mat i_training_data, CvPoint2D32f i_LVQ_weight[4] ,int i_Epoch, float i_learning_rate)
{
	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	char num1[200], num2[200], num3[200], path0[100];
	i_show_result = Mat::zeros(1000, 1000, CV_8UC3);
	for (int i = 0; i < 9; i++)
	{
		for (int j = 0; j < 9; j++)
		{
			switch ((int)i_training_data.at<int>(i, j))
			{
			case 0:
				line(i_show_result, Point((j + 1) * 100, (i + 1) * 100), Point((j + 1) * 100, (i + 1) * 100), Scalar(0, 0, 255), 20, 8);
				break;
			case 1:
				line(i_show_result, Point((j + 1) * 100, (i + 1) * 100), Point((j + 1) * 100, (i + 1) * 100), Scalar(0, 255, 0), 20, 8);
				break;
			case 2:
				line(i_show_result, Point((j + 1) * 100, (i + 1) * 100), Point((j + 1) * 100, (i + 1) * 100), Scalar(255, 255, 255), 20, 8);
				break;
			case 3:
				line(i_show_result, Point((j + 1) * 100, (i + 1) * 100), Point((j + 1) * 100, (i + 1) * 100), Scalar(255, 0, 0), 20, 8);
				break;

			default:
				break;
			}
		}
	}
	circle(i_show_result, Point(i_LVQ_weight[0].x * 1000, i_LVQ_weight[0].y * 1000), 30, Scalar(0, 0, 255), 5, 8);
	circle(i_show_result, Point(i_LVQ_weight[1].x * 1000, i_LVQ_weight[1].y * 1000), 30, Scalar(0, 255, 0), 5, 8);
	circle(i_show_result, Point(i_LVQ_weight[2].x * 1000, i_LVQ_weight[2].y * 1000), 30, Scalar(255, 255, 255), 5, 8);
	circle(i_show_result, Point(i_LVQ_weight[3].x * 1000, i_LVQ_weight[3].y * 1000), 30, Scalar(255, 0, 0), 5, 8);


	sprintf_s(num1, "Epoch = %d", i_Epoch);
	cv::putText(i_show_result, num1, Point(0, 30), 0, 1, Scalar(255, 255, 255), 2);
	sprintf_s(num2, "learning_rate = %f", i_learning_rate);
	cv::putText(i_show_result, num2, Point(0, 70), 0, 1, Scalar(255, 255, 255), 2);
	sprintf_s(path0, "photo%d.png", i_Epoch);
	imwrite(path0, i_show_result, compression_params);
	imshow("result", i_show_result);
	waitKey(20);
	Sleep(500);
}
