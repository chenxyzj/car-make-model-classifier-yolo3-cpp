// Copyright © 2019 by Spectrico
// Licensed under the MIT License

#include <iostream>
#include <fstream>
#include <numeric>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

const double confThreshold = 0.3;
const double nmsThreshold = 0.3;

template <typename T>
std::vector<size_t> SortIndexes(const std::vector<T> &v) {

	// initialize original index locations
	std::vector<size_t> idx(v.size());
	std::iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	std::sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] > v[i2]; });

	return idx;
}

std::vector<std::string> readClassNames(std::string filename)
{
	std::vector<std::string> classNames;

	std::ifstream fp(filename);
	if (!fp.is_open())
	{
		std::cerr << "File with classes labels not found: " << filename << std::endl;
		exit(-1);
	}

	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}

	fp.close();
	return classNames;
}

void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs, cv::dnn::Net *pnet, std::vector<int> &classIds, std::vector<float> &confidences, std::vector<cv::Rect> &boxes, std::vector<int> &indices)
{
	static std::vector<int> outLayers = pnet->getUnconnectedOutLayers();
	static std::string outLayerType = pnet->getLayer(outLayers[0])->type;

	if (outLayerType == "DetectionOutput")
	{
		// Network produces output blob with a shape 1x1xNx7 where N is a number of
		// detections and an every detection is a vector of values
		// [batchId, classId, confidence, left, top, right, bottom]
		CV_Assert(outs.size() > 0);
		for (size_t k = 0; k < outs.size(); k++)
		{
			float* data = (float*)outs[k].data;
			for (size_t i = 0; i < outs[k].total(); i += 7)
			{
				float confidence = data[i + 2];
				if (confidence > confThreshold)
				{
					int left = (int)data[i + 3];
					int top = (int)data[i + 4];
					int right = (int)data[i + 5];
					int bottom = (int)data[i + 6];
					int width = right - left + 1;
					int height = bottom - top + 1;
					if (width * height <= 1)
					{
						left = (int)(data[i + 3] * frame.cols);
						top = (int)(data[i + 4] * frame.rows);
						right = (int)(data[i + 5] * frame.cols);
						bottom = (int)(data[i + 6] * frame.rows);
						width = right - left + 1;
						height = bottom - top + 1;
					}
					classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
					boxes.push_back(cv::Rect(left, top, width, height));
					confidences.push_back(confidence);
				}
			}
		}
	}
	else if (outLayerType == "Region")
	{
		for (size_t i = 0; i < outs.size(); ++i)
		{
			// Network produces output blob with a shape NxC where N is a number of
			// detected objects and C is a number of classes + 4 where the first 4
			// numbers are [center_x, center_y, width, height]
			float* data = (float*)outs[i].data;
			for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
			{
				cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
				cv::Point classIdPoint;
				double confidence;
				cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
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
					boxes.push_back(cv::Rect(left, top, width, height));
				}
			}
		}
	}
	else
		CV_Error(cv::Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);
}

cv::Mat getPaddedROI(const cv::Mat &input, int top_left_x, int top_left_y, int width, int height, cv::Scalar paddingColor) {
	int bottom_right_x = top_left_x + width;
	int bottom_right_y = top_left_y + height;

	cv::Mat output;
	if (top_left_x < 0 || top_left_y < 0 || bottom_right_x > input.cols || bottom_right_y > input.rows) {
		// border padding will be required
		int border_left = 0, border_right = 0, border_top = 0, border_bottom = 0;

		if (top_left_x < 0) {
			width = width + top_left_x;
			border_left = -1 * top_left_x;
			top_left_x = 0;
		}
		if (top_left_y < 0) {
			height = height + top_left_y;
			border_top = -1 * top_left_y;
			top_left_y = 0;
		}
		if (bottom_right_x > input.cols) {
			width = width - (bottom_right_x - input.cols);
			border_right = bottom_right_x - input.cols;
		}
		if (bottom_right_y > input.rows) {
			height = height - (bottom_right_y - input.rows);
			border_bottom = bottom_right_y - input.rows;
		}

		cv::Rect R(top_left_x, top_left_y, width, height);
		copyMakeBorder(input(R), output, border_top, border_bottom, border_left, border_right, cv::BORDER_CONSTANT, paddingColor);
	}
	else {
		// no border padding required
		cv::Rect R(top_left_x, top_left_y, width, height);
		output = input(R);
	}
	return output;
}

cv::Mat GetSquareImage(const cv::Mat& img, int target_width)
{
	int width = img.cols,
		height = img.rows;

	cv::Mat square = cv::Mat::zeros(target_width, target_width, img.type());

	int max_dim = (width >= height) ? width : height;
	double scale = ((double)target_width) / max_dim;
	cv::Rect roi;
	if (width >= height)
	{
		roi.width = target_width;
		roi.x = 0;
		roi.height = height * scale;
		roi.y = (target_width - roi.height) / 2;
	}
	else
	{
		roi.y = 0;
		roi.height = target_width;
		roi.width = width * scale;
		roi.x = (target_width - roi.width) / 2;
	}

	cv::resize(img, square(roi), roi.size());

	return square;
}

int main(int argc, char** argv)
{
	cv::dnn::Net net_car_make_model_classifier;
	//const std::string modelFile = "model-weights-spectrico-mmr-mobilenet-224x224-908A6A8C.pb";
	const std::string modelFile = "model-weights-spectrico-mmr-mobilenet-64x64-531A7126.pb";
	//const int classifier_input_size = 224;
	const int classifier_input_size = 64;
	//! [Initialize network]
	net_car_make_model_classifier = cv::dnn::readNetFromTensorflow(modelFile);
	if (net_car_make_model_classifier.empty())
	{
		std::cerr << "Can't load network by using the model file: " << std::endl;
		std::cerr << modelFile << std::endl;
		exit(-1);
	}

	std::vector<std::string> classNamesCarMakeModelClassifier = readClassNames("labels.txt");

	std::string modelObjectDetector = "yolov3.weights";
	std::string modelObjectDetectorConfig = "yolov3.cfg";
	// Load the object detection model
	cv::dnn::Net net_object_detector = cv::dnn::readNet(modelObjectDetector, modelObjectDetectorConfig);
	if (net_object_detector.empty())
	{
		std::cerr << "Can't load network by using the model file: " << std::endl;
		std::cerr << modelObjectDetector << std::endl;
		exit(-1);
	}
	net_object_detector.setPreferableBackend(0);
	net_object_detector.setPreferableTarget(0);
	std::vector<cv::String> outNamesObjectDetector;
	outNamesObjectDetector = net_object_detector.getUnconnectedOutLayersNames();


	std::string imageFile = argc == 2 ? argv[1] : "cars.jpg";
	//! [Prepare blob]
	cv::Mat img = cv::imread(imageFile, cv::IMREAD_COLOR);
	if (img.empty() || !img.data)
	{
		std::cerr << "Can't read image from the file: " << imageFile << std::endl;
		exit(-1);
	}

	cv::Mat blob;
	cv::dnn::blobFromImage(img, blob, 1 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
	// Run a model.
	net_object_detector.setInput(blob);
	std::vector<cv::Mat> outs;
	net_object_detector.forward(outs, outNamesObjectDetector);

	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<int> indices;

	postprocess(img, outs, &net_object_detector, classIds, confidences, boxes, indices);

	cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		cv::Rect box = boxes[idx];
		float confidence = confidences[idx];

		int classIdObjectDetector = classIds[idx];
		if (!(classIdObjectDetector == 3 - 1) || (classIdObjectDetector == 6 - 1) || (classIdObjectDetector == 8 - 1))
			continue;

		std::string inBlobName = "input_1";
		std::string outBlobName = "softmax/Softmax";
		cv::Mat img_car = getPaddedROI(img, box.x, box.y, box.width, box.height, cv::BORDER_REPLICATE);
		img_car = GetSquareImage(img_car, classifier_input_size);

		cv::Mat inputBlob = cv::dnn::blobFromImage(img_car, 0.0078431372549019607843137254902, cv::Size(classifier_input_size, classifier_input_size), cv::Scalar(127.5, 127.5, 127.5), true, false, CV_32F);   //Convert Mat to image batch
		net_car_make_model_classifier.setInput(inputBlob, inBlobName);        //set the network input
		cv::TickMeter tm;
		tm.start();
		cv::Mat resultCarMakeModelClassifier = net_car_make_model_classifier.forward(outBlobName);                          //compute output
		tm.stop();
		std::cout << "------------------------------------------------" << std::endl;
		std::cout << "Object box: " << box << std::endl;
		std::cout << "Inference time, ms: " << tm.getTimeMilli() << std::endl;
		const int top_n = 3;
		std::cout << "Top " << std::to_string(top_n) << " probabilities: " << std::endl;
		cv::Mat probMat = resultCarMakeModelClassifier.reshape(1, 1);
		std::vector<float>vec(probMat.begin<float>(), probMat.end<float>());
		int top = 0;
		for (auto i : SortIndexes(vec))
		{
			std::string make_model = classNamesCarMakeModelClassifier.at(i);
			std::string make;
			std::string model;
			size_t pos = 0;
			pos = make_model.find("\t");
			if (pos != std::string::npos)
			{
				make = make_model.substr(0, pos);
				model = make_model.substr(pos + 1);
			}

			std::cout << "make: " << make << "\tmodel: " << model << "\tconfidence: " << vec[i] * 100 << " %" << std::endl;
			if (++top == top_n)
				break;
		}
		std::cout << "------------------------------------------------" << std::endl;
		std::cout << std::endl;
	}

	return 0;
}
