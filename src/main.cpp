#include "MTCNN.h"
#include "mrdir.h"
#include "mropencv.h"
#include "mrutil.h"
using namespace mtcnn;
using namespace std;
const std::string rootdir = "../";
const std::string imgdir = rootdir+"/imgs";
const std::string resultdir = rootdir + "/results";
const std::string proto_model_dir = rootdir + "/model";

#if _WIN32
const string casiadir = "E:/Face/Datasets/CASIA-maxpy-clean";
const string outdir = "E:/Face/Datasets/CASIA-mtcnn";
#else
const string casiadir = "~\CASIA-maxpy-clean";
const string outdir = "~\CASIA-mtcnn";
#endif

int testcamera(int cameraindex=0)
{
	double threshold[3] = { 0.6, 0.7, 0.5 };
	double factor = 0.709;
	int minSize = 50;
	MTCNN detector(proto_model_dir);
	cv::VideoCapture cap(cameraindex);
	cv::Mat frame;
	while (cap.read(frame)){
		std::vector<FaceInfo> faceInfo;
		TickMeter tm;
		tm.start();
		detector.Detect(frame, faceInfo, minSize, threshold, factor);
		tm.stop();
		cout << tm.getTimeMilli() << "ms" << endl;
		MTCNN::drawDectionResult(frame, faceInfo);
		cv::imshow("img", frame);
		if ((char)cv::waitKey(1) == 'q')
			break;
	}
	return 0;
}

vector<cv::Mat> Align5points(const cv::Mat &img, const std::vector<FaceInfo>&faceInfo)
{
	std::vector<cv::Point2f>  p2s;
	p2s.push_back(cv::Point2f(30.2946, 51.6963));
	p2s.push_back(cv::Point2f(65.5318, 51.5014));
	p2s.push_back(cv::Point2f(48.0252, 71.7366));
	p2s.push_back(cv::Point2f(33.5493, 92.3655));
	p2s.push_back(cv::Point2f(62.7299, 92.2041));
	vector<Mat>dsts;
	for (int i = 0; i < faceInfo.size(); i++)
	{
		std::vector<cv::Point2f> p1s;
		FacePts facePts = faceInfo[i].facePts;
		for (int j = 0; j < 5; j++)
		{
			p1s.push_back(cv::Point(facePts.y[j], facePts.x[j]));
		}
		cv::Mat t = cv::estimateRigidTransform(p1s, p2s, false);
		if (!t.empty())
		{
			Mat dst;
			cv::warpAffine(img, dst, t, Size(96, 112));
			dsts.push_back(dst);
		}
		else
		{
			dsts.push_back(img);
		}
	}
	return dsts;
}

int testdir()
{
	double threshold[3] = { 0.6, 0.7, 0.5 };
	double factor = 0.709;
	int minSize = 50;
	MTCNN detector(proto_model_dir);
	vector<string>files=getAllFilesinDir(imgdir);
	cv::Mat frame;

	for (int i = 0; i < files.size(); i++)
	{
		string imageName = imgdir + "/" + files[i];
		frame=cv::imread(imageName);
		clock_t t1 = clock();
		std::vector<FaceInfo> faceInfo;
		detector.Detect(frame, faceInfo, minSize, threshold, factor);
		std::cout << "Detect Time: " << (clock() - t1)*1.0 / 1000 << std::endl;
		vector<Mat> alignehdfaces = Align5points(frame,faceInfo);
		for (int j = 0; j < alignehdfaces.size(); j++)
		{
			string alignpath="align/"+int2string(j)+"_"+files[i];
			imwrite(alignpath, alignehdfaces[j]);
		}
		MTCNN::drawDectionResult(frame,faceInfo);
		cv::imshow("img", frame);
		string resultpath = resultdir + "/"+files[i];
		cv::imwrite(resultpath, frame);
		cv::waitKey();
	}
	return 0;
}

int testibm()
{
	double threshold[3] = { 0.6, 0.7, 0.5 };
	double factor = 0.709;
	int minSize = 50;
	MTCNN detector(proto_model_dir);
	vector<string>files=getAllFilesinDir(imgdir);
	cv::Mat frame;
	for (int i = 0; i < files.size(); i++)
	{
		string imageName = imgdir + "/" + files[i];
		frame = cv::imread(imageName);
		clock_t t1 = clock();
		std::vector<FaceInfo> faceInfo;
		detector.Detect(frame, faceInfo, minSize, threshold, factor);
		std::cout << "Detect Using: " << (clock() - t1)*1.0 / 1000 << std::endl;
		MTCNN::drawDectionResult(frame, faceInfo);
		cv::imshow("img", frame);
		string resultpath = resultdir + "/" + files[i];
		cv::imwrite(resultpath, frame);
		cv::waitKey(1);
	}
	return 0;
}
#define SAVE_FDDB_RESULTS 1
int eval_fddb()
{
	const char* fddb_dir = "E:/Face/Datasets/fddb";
	string format = fddb_dir + string("/MTCNN/%Y%m%d-%H%M%S");
	time_t t = time(NULL);
	char buff[300];
	strftime(buff, sizeof(buff), format.c_str(), localtime(&t));
	if (SAVE_FDDB_RESULTS) {
		MKDIR(buff);
	}
	string result_prefix(buff);
	string prefix = fddb_dir + string("/images/");
	double threshold[3] = { 0.6, 0.7, 0.5 };
	double factor = 0.709;
	int minSize = 50;
	MTCNN detector(proto_model_dir);
	int counter = 0;
//#pragma omp parallel for
	for (int i = 1; i <= 10; i++) 
	{
		char fddb[300];
		char fddb_out[300];
		char fddb_answer[300];
		cout<<"Folds: "<<i<<endl;
		sprintf(fddb, "%s/FDDB-folds/FDDB-fold-%02d.txt", fddb_dir, i);
		sprintf(fddb_out, "%s/MTCNN/fold-%02d-out.txt", fddb_dir, i);
		sprintf(fddb_answer, "%s/FDDB-folds/FDDB-fold-%02d-ellipseList.txt", fddb_dir, i);

		FILE* fin = fopen(fddb, "r");
		FILE* fanswer = fopen(fddb_answer, "r");
#ifdef _WIN32
		FILE* fout = fopen(fddb_out, "wb"); // replace \r\n on Windows platform		
#else
		FILE* fout = fopen(fddb_out, "w");	
#endif // WIN32
		
		char path[300];
		int counter = 0;
		while (fscanf(fin, "%s", path) > 0)
		{
			string full_path = prefix + string(path) + string(".jpg");
			Mat img = imread(full_path);
			if (!img.data) {
				cout << "Cannot read " << full_path << endl;;
				continue;
			}
			clock_t t1 = clock();
			std::vector<FaceInfo> faceInfo;
			detector.Detect(img, faceInfo, minSize, threshold, factor);
			std::cout << "Detect " <<i<<": "<<counter<<" Using : " << (clock() - t1)*1.0 / 1000 << std::endl;
			const int n = faceInfo.size();
			fprintf(fout, "%s\n%d\n", path, n);
			for (int j = 0; j < n; j++) {
				int x = (int)faceInfo[j].bbox.x1;
				if (x < 0)x = 0;
				int y = (int)faceInfo[j].bbox.y1;
				if (y < 0)y = 0;
				int h = (int)faceInfo[j].bbox.x2 - faceInfo[j].bbox.x1 + 1;
				if (h>img.rows - x)h = img.rows - x;
				int w = (int)faceInfo[j].bbox.y2 - faceInfo[j].bbox.y1 + 1;
				if (w>img.cols-y)w = img.cols - y;
				float score = faceInfo[j].bbox.score;
				cv::rectangle(img, cv::Rect(y, x, w, h), cv::Scalar(0, 0, 255), 1);
				fprintf(fout, "%d %d %d %d %lf\n", y, x, w, h, score);
			}
			for (int t = 0; t < faceInfo.size(); t++){
				FacePts facePts = faceInfo[t].facePts;
				for (int j = 0; j < 5; j++)
					cv::circle(img, cv::Point(facePts.y[j], facePts.x[j]), 1, cv::Scalar(255, 255, 0), 2);
			}
			cv::imshow("img", img);
			cv::waitKey(1);
			char buff[300];
			if (SAVE_FDDB_RESULTS) {
				counter++;
				sprintf(buff, "%s/%02d_%03d.jpg", result_prefix.c_str(), i, counter);
				// get answer
				int face_n = 0;
				fscanf(fanswer, "%s", path);
				fscanf(fanswer, "%d", &face_n);
				for (int k = 0; k < face_n; k++)
				{
					double major_axis_radius, minor_axis_radius, angle, center_x, center_y, score;
					fscanf(fanswer, "%lf %lf %lf %lf %lf %lf", &major_axis_radius, &minor_axis_radius, \
						&angle, &center_x, &center_y, &score);
					// draw answer
					angle = angle / 3.1415926*180.;
					cv::ellipse(img, Point2d(center_x, center_y), Size(major_axis_radius, minor_axis_radius), \
						angle, 0., 360., Scalar(255, 0, 0), 2);					
				}
				cv::imwrite(buff, img);
			}
		}
		fclose(fin);
		fclose(fout);
		fclose(fanswer);
	}
	return 0;
}

int extractCASIA()
{
	::google::InitGoogleLogging("");
	double threshold[3] = { 0.6, 0.7, 0.5 };
	double factor = 0.709;
	int minSize = 50;
	MTCNN detector(proto_model_dir);
	vector<string>subdirs=getAllSubdirs(casiadir);
	MKDIR(outdir.c_str());
	for (int i =0; i < subdirs.size(); i++)
	{
		string subdir = casiadir + "/" + subdirs[i];
		vector<string>files=getAllFilesinDir(subdir);
		string outsubdir = outdir + "/" + subdirs[i];
		MKDIR(outsubdir.c_str());
		for (int j = 0; j < files.size(); j++)
		{
			cout << i <<":"<<subdirs[i]<< " " << j << endl;
			string filepath = subdir + "/" + files[j];
			std::vector<FaceInfo> faceInfo;
			Mat frame = imread(filepath);
			detector.Detect(frame, faceInfo, minSize, threshold, factor);
			if (faceInfo.size()>0)
			{
				int maxindex = 0, maxarea = 0;
				for (int k = 0; k < faceInfo.size(); k++)
				{
					auto bbox = faceInfo[k].bbox;
					int area = (bbox.x2 - bbox.x1)*(bbox.x2 - bbox.x1) + (bbox.y2 - bbox.y1)*(bbox.y2 - bbox.y1);
					if (area>maxarea)
					{
						maxarea = area;
						maxindex = k;
					}
				}
				vector<FaceInfo>maxface;
				maxface.push_back(faceInfo[maxindex]);
				vector<Mat> alignedfaces = Align5points(frame, maxface);
				if (alignedfaces.size() > 0)
				{
					string outpath = outsubdir + "/" + files[j];
					imwrite(outpath, alignedfaces[0]);
				}
				else
				{
					cout << "No Aligend Face" << endl;
				}
			}
			else
			{
				cout << "No Faces detected" << endl;
			}
		}
	}
	return 0;
}
int main(int argc, char **argv)
{
//    testcamera();
	testdir();
//	testibm();
//	eval_fddb();
//	extractCASIA();
}