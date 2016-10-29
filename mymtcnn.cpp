#include <cassert>
#include "mropencv.h"
#include <caffe/caffe.hpp>

using namespace cv;
using namespace std;
using namespace caffe;

const string modeldir = "model";

caffe::Net<float> *PNet;
caffe::Net<float> *RNet;
caffe::Net<float> *ONet;

void initmodel(const string modeldir="model")
{
	string prototxtpath = modeldir + "/det1.prototxt";
	string modelpath = modeldir + "/det1.caffemodel";
	PNet = new caffe::Net<float>(prototxtpath, caffe::TEST);
	PNet->CopyTrainedLayersFrom(modelpath);

	prototxtpath = modeldir + "/det2.prototxt";
	modelpath = modeldir + "/det2.caffemodel";
	RNet = new caffe::Net<float>(prototxtpath, caffe::TEST);
	RNet->CopyTrainedLayersFrom(modelpath);

	prototxtpath = modeldir + "/det3.prototxt";
	modelpath = modeldir + "/det3.caffemodel";
	ONet = new caffe::Net<float>(prototxtpath, caffe::TEST);
	ONet->CopyTrainedLayersFrom(modelpath);
}

class detInfo
{
	cv::Rect rect_;
	std::vector<cv::Point> pts_;
};



vector<detInfo> detectface(cv::Mat &src, int minsize, caffe::Net<float>*PNet, caffe::Net<float>*RNet, caffe::Net<float>*ONet, std::vector<float> thresholds, bool fastresize = false, float factor = 0.709)
{
	int factor_count = 0;
	vector<detInfo> dets;
	int h = src.rows;
	int w = src.cols;
	int minl = min(w, h);
	Mat img;
	src.convertTo(img,CV_32FC3);
	vector<float>scales;
	float m = 12.0 / minsize;
	minl = minl*m;
	while (minl>=12)
	{
		float scale = m*pow(factor,factor_count);
		scales.push_back(scale);
		minl = minl*factor;
		factor_count += 1;
	}
	for (int j = 0; j < scales.size(); j++)
	{
		float scale = scales[j];
		int hs = ceil(h*scale);
		int ws = ceil(w*scale);
		Mat im_data=src.clone();
		im_data.convertTo(im_data, CV_64FC3);
		resize(im_data, im_data, Size(hs, ws), CV_INTER_LINEAR);
		cv::Scalar mean = { 127.5, 127.5, 127.5};
		im_data -= mean;
		im_data *=0.0078125;
		PNet->blob_by_name("data")->Reshape(1,3,hs,ws);
		PNet->Reshape();

		std::vector<cv::Mat> input_channels;
		Blob<float>* input_layer = PNet->input_blobs()[0];
		int width = input_layer->width();
		int height = input_layer->height();
		float* input_data = input_layer->mutable_cpu_data();
		for (int i = 0; i < input_layer->channels(); ++i) {
			cv::Mat channel(height, width, CV_32FC1, input_data);
			input_channels.push_back(channel);
			input_data += width * height;
		}
		cv::split(im_data, input_channels);
		cv::Mat i0 = input_channels[0];
		Mat i1 = input_channels[1];
		Mat i2 = input_channels[2];
		auto out=PNet->Forward();
		auto output_layer = PNet->output_blobs()[0];
		const float* begin = output_layer->cpu_data();
		const float* end = begin + output_layer->count();
		for (auto it = begin; it != end; it++)
			cout << *it << " ";
	}
	return dets;
}
int main(int argc, char *argv[])
{
//	caffe::GlobalInit(&argc, &argv);
	Caffe::set_mode(Caffe::GPU);
	initmodel();
	cv::VideoCapture capture(0);
	cv::Mat frame;
	vector<float> thresholds;
	thresholds.push_back(0.7);
	thresholds.push_back(0.6);
	thresholds.push_back(0.5);
	frame = cv::imread("Jennifer_Aniston_0016.jpg");
	detectface(frame, 50, PNet, RNet, ONet, thresholds);
// 	while (1)
// 	{
// 		capture >> frame;
// 		if (!frame.data)
// 			break;
// 		detectface(frame, 50, PNet, RNet, ONet, thresholds);
// 		cv::imshow("img", frame);
// 		cv::waitKey(1);
// 	}
  return 0;
}
