//纯C++超分辨率重建DRRN --改编--（三）主函数 DRRN
//https://blog.csdn.net/juebai123/article/details/83150984

// 超分辨率重建（卷积神经网络(DRRN)）
//
// 设定参数：文件名、放大倍数

#include <conio.h>

#include <math.h>

#include <iostream>
#include <fstream>

#include <vector>

#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include <samples/ocv_common.hpp>


// [Prologue]
#include "mkldnn.hpp"

// Optional header to access debug functions like `mkldnn_status2str()`
#include "mkldnn_debug.h"

using namespace mkldnn;

using namespace std;

#define TOTAL_LAYER 8


struct LayerData
{
	char type[256];
	int layer_number;
	int weights_len;
	float *	weights_data;
	int bias_len;
	float *	bias_data;
	//int prelu_alpha_len;
	//float *	prelu_alpha_data;
	int input_dimension;
	int output_dimension;
	int kernel_size;
};


struct basic_unit////也就是 残差块的半LayerData
{
	LayerData *conv_layer;
	int normalize_len;
	float *	u; //均值 和方差
	float *	std;
	float *	alpha; //缩放 和位移
	float *	beta;
};

struct residual_block
{
	int num;//数量 2
	basic_unit *data;
};


struct DRRN_Model  //不分倍数，通用
{


	basic_unit the_first_unit; //半层

	int residual_block_number;//9块
	residual_block * block;

	basic_unit the_last_unit; //总10层

			  //构造函数
	DRRN_Model();

};


DRRN_Model::DRRN_Model()
{

	int size;
	int len;
	size = sizeof(LayerData);//
	the_first_unit.conv_layer = (LayerData *)malloc(size);
	LayerData *current_layer = the_first_unit.conv_layer;
	strcpy(current_layer->type, "conv_layer");
	current_layer->layer_number = 1;
	current_layer->weights_len = 128 * 1 * 3 * 3;
	current_layer->weights_data = (float*)malloc(sizeof(float) * current_layer->weights_len);
	current_layer->input_dimension = 1;
	current_layer->output_dimension = 128;
	current_layer->kernel_size = 3;
	current_layer->bias_len = 128;
	current_layer->bias_data = (float*)malloc(sizeof(float) * current_layer->bias_len);

	the_first_unit.normalize_len = len = 1;
	the_first_unit.u = (float*)malloc(sizeof(float) * len);
	the_first_unit.std = (float*)malloc(sizeof(float) * len);
	the_first_unit.alpha = (float*)malloc(sizeof(float) * len);
	the_first_unit.beta = (float*)malloc(sizeof(float) * len);

	//块 
	residual_block_number = 9;
	size = sizeof(residual_block)*residual_block_number;
	block = (residual_block*)malloc(size);
	residual_block * residual_block0 = block;
	//----------所有残差块卷积核重用一套数据
	float *	the_first_weights_data;
	float *	the_first_bias_data;
	float *	the_second_weights_data;
	float *	the_second_bias_data;
	//-------------
	for (int k = 0; k<residual_block_number; k++)
	{
		//先设置"残差块"

		residual_block0->num = 2;

		size = sizeof(basic_unit) * 2;

		residual_block0->data = (basic_unit*)malloc(size);

		for (int p = 0; p<2; p++) {
			size = sizeof(LayerData);//
			residual_block0->data[p].conv_layer = (LayerData *)malloc(size);
			current_layer = residual_block0->data[p].conv_layer;
			strcpy(current_layer->type, "conv_layer");
			current_layer->weights_len = 128 * 128 * 3 * 3;
			if (k == 0) {
				current_layer->weights_data = (float*)malloc(sizeof(float) * current_layer->weights_len);
				if (p == 0)
					the_first_weights_data = current_layer->weights_data;
				else
					the_second_weights_data = current_layer->weights_data;
			}
			else
			{
				if (p == 0)
					current_layer->weights_data = the_first_weights_data;
				else
					current_layer->weights_data = the_second_weights_data;
			}
			current_layer->input_dimension = 128;
			current_layer->output_dimension = 128;
			current_layer->kernel_size = 3;
			current_layer->bias_len = 128;
			if (k == 0) {
				current_layer->bias_data = (float*)malloc(sizeof(float) * current_layer->bias_len);
				if (p == 0)
					the_first_bias_data = current_layer->bias_data;
				else
					the_second_bias_data = current_layer->bias_data;
			}
			else
			{
				if (p == 0)
					current_layer->bias_data = the_first_bias_data;
				else
					current_layer->bias_data = the_second_bias_data;
			}


			residual_block0->data[p].normalize_len = len = 128;
			residual_block0->data[p].u = (float*)malloc(sizeof(float) * len);
			residual_block0->data[p].std = (float*)malloc(sizeof(float) * len);
			residual_block0->data[p].alpha = (float*)malloc(sizeof(float) * len);
			residual_block0->data[p].beta = (float*)malloc(sizeof(float) * len);
		}

		residual_block0++;
	}


	size = sizeof(LayerData);//
	the_last_unit.conv_layer = (LayerData *)malloc(size);
	current_layer = the_last_unit.conv_layer;
	strcpy(current_layer->type, "conv_layer");
	current_layer->weights_len = 1 * 128 * 3 * 3;
	current_layer->weights_data = (float*)malloc(sizeof(float) * current_layer->weights_len);
	current_layer->input_dimension = 128;
	current_layer->output_dimension = 1;
	current_layer->kernel_size = 3;
	current_layer->bias_len = 1;
	current_layer->bias_data = (float*)malloc(sizeof(float) * current_layer->bias_len);

	the_last_unit.normalize_len = 128;
	the_last_unit.u = (float*)malloc(sizeof(float) * the_last_unit.normalize_len);
	the_last_unit.std = (float*)malloc(sizeof(float) * the_last_unit.normalize_len);
	the_last_unit.alpha = (float*)malloc(sizeof(float) * the_last_unit.normalize_len);
	the_last_unit.beta = (float*)malloc(sizeof(float) * the_last_unit.normalize_len);



}

void locate_data_blobs(std::ifstream &fin)
{
	char  line0[256];       //每次从文件读一行 
	while (!fin.eof())
	{
		fin.getline(line0, 255);
		//cout<<line0<<endl;

		//char *strstr(const char *haystack, const char *needle)
		//在字符串 haystack 中查找第一次出现字符串 needle（不包含空结束字符）的位置。
		if (strstr(line0, "blobs"))
			break;
	}
}

char* locate_blobs(char * keyword, std::ifstream &fin)
{
	char  line0[256];       //每次从文件读一行 
	while (!fin.eof())
	{
		fin.getline(line0, 255);
		//cout<<line0<<endl;

		//char *strstr(const char *haystack, const char *needle)
		//在字符串 haystack 中查找第一次出现字符串 needle（不包含空结束字符）的位置。
		if (strstr(line0, keyword))
		{
			//cout << line0 << endl;
			return line0;
		}
	}
	cout << "EOF" << endl;
}

bool parseModel(DRRN_Model *sr)
{
	char name[] = "c:\\work\\fsrcnn\\DRRN_B1U9_20C128_iter_464056.caffemodel.txt";
	std::ifstream fin(name);

	//检查文件是否存在
	if (!fin)
	{
		return false;
	}

	cout << "loading DRRN_B1U9_20C128_iter_464056.txt ..." << endl;


	//从档案载入
	char str[40];
	int len;
	float *	data;
	//for (int k = 0; k < 94; k++)
	//{
	//	cout << "layer: " << k << endl;
	//	cout << locate_blobs("layer", fin) << endl;
	//	cout << locate_blobs("name",fin) << endl;
	//	cout << locate_blobs("type", fin) << endl;
	//}

	float dummy;
	for (int k = 0; k < 20; k++)
	{
		cout << "layer: " << k << endl;
		cout << locate_blobs("Convolution", fin) << endl;
		locate_data_blobs(fin);
		for (int i = 0; i<5; i++)
		{
			fin >> str;
			fin >> dummy;
			cout << dummy << endl;
		}
		cout << "****************" << endl;
		for (int i = 0; i<5; i++)
		{
			fin >> str;
			fin >> dummy;
			cout << dummy << endl;
		}
		cout << "****************" << endl;
	}

	fin.close();
	return true;
}

bool loadModel(DRRN_Model *sr)
{
	char name[] = "c:\\work\\fsrcnn\\DRRN_B1U9_20C128_iter_464056.caffemodel.txt";
	std::ifstream fin(name);

	//检查文件是否存在
	if (!fin)
	{
		return false;
	}

	cout << "loading DRRN_B1U9_20C128_iter_464056.txt ..." << endl;


	//从档案载入
	char str[40];
	char * temp_str;
	int len;
	float *	data, scale_factor;
	int i, j, k;

	//第一层
	LayerData * current_layer = sr->the_first_unit.conv_layer;
	cout << "the first layer" << endl;
	/*********************************************/
	temp_str = locate_blobs("BatchNorm", fin);
	cout << temp_str << endl;
	locate_data_blobs(fin);
	data = sr->the_first_unit.u;
	for (i = 0; i < sr->the_first_unit.normalize_len; i++)
	{
		fin >> str;
		fin >> *data++;
	}
	data = sr->the_first_unit.std;
	locate_data_blobs(fin);
	for (i = 0; i < sr->the_first_unit.normalize_len; i++)
	{
		fin >> str;
		fin >> *data++;
	}
	locate_data_blobs(fin);
	fin >> str;
	fin >> scale_factor;

	data = sr->the_first_unit.u;
	for (i = 0; i < sr->the_first_unit.normalize_len; i++)
	{
		*data = (*data) / scale_factor;;
		data++;
	}
	data = sr->the_first_unit.std;
	for (i = 0; i < sr->the_first_unit.normalize_len; i++)
	{
		*data = ((*data) / scale_factor);
		data++;
	}
	/*********************************************/
	temp_str = locate_blobs("Scale", fin);
	cout << temp_str << endl;
	locate_data_blobs(fin);
	data = sr->the_first_unit.alpha;
	for (i = 0; i < sr->the_first_unit.normalize_len; i++)
	{
		fin >> str;
		fin >> *data++;
	}
	data = sr->the_first_unit.beta;
	locate_data_blobs(fin);
	for (i = 0; i < sr->the_first_unit.normalize_len; i++)
	{
		fin >> str;
		fin >> *data++;
	}
	/*********************************************/
	temp_str = locate_blobs("Convolution", fin);
	cout << temp_str << endl;
	//先到数据位置 blobs
	locate_data_blobs(fin);
	//1。读入权重
	len = current_layer->weights_len;//需要载入的个数
	data = current_layer->weights_data;

	//float tmp;
	for (int i = 0; i<len; i++)
	{
		fin >> str;
		fin >> *data++;
	}
	//2。读入偏置
	locate_data_blobs(fin);
	len = current_layer->bias_len;//需要载入的个数
	data = current_layer->bias_data;
	for (int i = 0; i<len; i++)
	{
		fin >> str;
		fin >> *data++;
	}
	/*********************************************/

	//中间9层 （18block)
	residual_block * residual_block0 = sr->block;
	for (int k = 0; k<sr->residual_block_number; k++)  //9个残差快
	{
		cout<<k<<"Layer"<<endl<<endl;
		for (int p = 0; p<2; p++) {
			cout << p << "sublayer" << endl;


			current_layer = residual_block0->data[p].conv_layer;
			/****************BatchNormal**********************/
			temp_str = locate_blobs("BatchNorm", fin);
			cout << temp_str << endl;
			locate_data_blobs(fin);
			data = residual_block0->data[p].u;
			for (i = 0; i < residual_block0->data[p].normalize_len; i++)
			{
				fin >> str;
				fin >> *data++;
			}
			data = residual_block0->data[p].std;
			locate_data_blobs(fin);
			for (i = 0; i < residual_block0->data[p].normalize_len; i++)
			{
				fin >> str;
				fin >> *data++;
			}
			locate_data_blobs(fin);
			fin >> str;
			fin >> scale_factor;

			data = residual_block0->data[p].u;
			for (i = 0; i < residual_block0->data[p].normalize_len; i++)
			{
				*data = (*data) / scale_factor;;
				data++;
			}
			data = residual_block0->data[p].std;
			for (i = 0; i < residual_block0->data[p].normalize_len; i++)
			{
				*data = ((*data) / scale_factor);
				data++;
			}
			/*************** Scale ***********************/
			temp_str = locate_blobs("Scale", fin);
			cout << temp_str << endl;
			locate_data_blobs(fin);
			data = residual_block0->data[p].alpha;
			for (i = 0; i < residual_block0->data[p].normalize_len; i++)
			{
				fin >> str;
				fin >> *data++;
			}
			data = residual_block0->data[p].beta;
			locate_data_blobs(fin);
			for (i = 0; i < residual_block0->data[p].normalize_len; i++)
			{
				fin >> str;
				fin >> *data++;
			}
			/*************** Convolution ************************/
			
			temp_str = locate_blobs("Convolution", fin);
			cout << temp_str << endl;
			//先到数据位置 blobs
			locate_data_blobs(fin);
			if (k == 0) //只取第一个residual_block的权重参数，后面8个block共用前面的
			{
				//1。读入权重
				len = current_layer->weights_len;//需要载入的个数
				data = current_layer->weights_data;

				//float tmp;
				for (int i = 0; i<len; i++)
				{
					fin >> str;
					fin >> *data++;
				}
				//2。读入偏置
				locate_data_blobs(fin);
				len = current_layer->bias_len;//需要载入的个数
				data = current_layer->bias_data;
				for (int i = 0; i<len; i++)
				{
					fin >> str;
					fin >> *data++;
				}
			}

		}
		residual_block0++;
	}
	//最后一层
	cout << "the Last Layer" << endl;
	current_layer = sr->the_last_unit.conv_layer;
	/*********************************************/
	temp_str = locate_blobs("BatchNorm", fin);
	cout << temp_str << endl;
	locate_data_blobs(fin);
	data = sr->the_last_unit.u;
	for (i = 0; i < sr->the_last_unit.normalize_len; i++)
	{
		fin >> str;
		fin >> *data++;
	}
	data = sr->the_last_unit.std;
	locate_data_blobs(fin);
	for (i = 0; i < sr->the_last_unit.normalize_len; i++)
	{
		fin >> str;
		fin >> *data++;
	}
	locate_data_blobs(fin);
	fin >> str;
	fin >> scale_factor;

	data = sr->the_last_unit.u;
	for (i = 0; i < sr->the_last_unit.normalize_len; i++)
	{
		*data = (*data) / scale_factor;;
		data++;
	}
	data = sr->the_last_unit.std;
	for (i = 0; i < sr->the_last_unit.normalize_len; i++)
	{
		*data = ((*data) / scale_factor);
		data++;
	}
	/*********************************************/
	temp_str = locate_blobs("Scale", fin);
	cout << temp_str << endl;
	locate_data_blobs(fin);
	data = sr->the_last_unit.alpha;
	for (i = 0; i < sr->the_last_unit.normalize_len; i++)
	{
		fin >> str;
		fin >> *data++;
	}
	data = sr->the_last_unit.beta;
	locate_data_blobs(fin);
	for (i = 0; i < sr->the_last_unit.normalize_len; i++)
	{
		fin >> str;
		fin >> *data++;
	}
	/*********************************************/
	temp_str = locate_blobs("Convolution", fin);
	cout << temp_str << endl;
	//先到数据位置 blobs
	locate_data_blobs(fin);
	//1。读入权重
	len = current_layer->weights_len;//需要载入的个数
	data = current_layer->weights_data;

	//float tmp;
	for (int i = 0; i<len; i++)
	{
		fin >> str;
		fin >> *data++;
	}
	//2。读入偏置
	locate_data_blobs(fin);
	len = current_layer->bias_len;//需要载入的个数
	data = current_layer->bias_data;
	for (int i = 0; i<len; i++)
	{
		fin >> str;
		fin >> *data++;
	}
	/*********************************************/


	fin.close();
	return true;
}


cv::Mat jpg;//一张原图

class ConvLayer
{
public:
	int		width;    //宽
	int     height;   //高
	int     depth;		  //通道 深度
	float * data;

	//构造函数
	ConvLayer(int iwidth, int iheight);
	ConvLayer(int iwidth, int iheight, int idepth);
	ConvLayer(int iwidth, int iheight, int c, float * data);
	~ConvLayer();
};

ConvLayer::ConvLayer(int iwidth, int iheight) : width(iwidth),
height(iheight)
{
	depth = 1;
	data = NULL;

}

ConvLayer::ConvLayer(int iwidth, int iheight, int idepth) : width(iwidth),
height(iheight), depth(idepth)
{
	data = NULL;

}
ConvLayer::ConvLayer(int iwidth, int iheight, int idepth, float * fdata) : width(iwidth),
height(iheight), depth(idepth), data(fdata)

{

}

ConvLayer::~ConvLayer()
{
	if (data != NULL)
	{
		delete []data;
		data = NULL;
	}
}

void loadjpg(char * jpgname)
{
	//loadimage(&jpg, jpgname);//
	jpg = cv::imread(jpgname);
	//resize to 640*480
	cv::resize(jpg, jpg, cv::Size(320, 240), 0, 0, cv::INTER_CUBIC);
}

//single color
ConvLayer Im2Matrix(cv::Mat *img)
{
	ConvLayer Matrix(img->cols, img->rows);
	Matrix.data = new float[img->cols * img->rows];
	unsigned char* pixels = (unsigned char*)(img->data);
	for (int i = 0; i < img->cols*img->rows; i++)
	{

		Matrix.data[i] = (float)(pixels[i]) / 255.0; //float [0:1]
	}
	return Matrix;
}

//single color
cv::Mat Matrix2Im(ConvLayer *Matrix)
{

	cv::Mat Img(Matrix->height, Matrix->width, CV_8U);	unsigned char *image_ptr = Img.data;	float dummy_1;	//std::memcpy(Img.data, c, 478 * 478 * sizeof(unsigned char));
	for (int i = 0; i < Matrix->width*Matrix->height; i++)
	{
		dummy_1 = Matrix->data[i] * 255.0; //float [0,255]
		if (dummy_1 < 0)
		{
			dummy_1 = 0;
		}
		else if (dummy_1 > 255)
		{
			dummy_1 = 255;
		}
		image_ptr[i] = (unsigned char)dummy_1;
	}
	return Img;

}


//BGR color
void BGRIm2Matrix(cv::Mat *imgBGR, ConvLayer *Y, ConvLayer *U, ConvLayer *V)
{
	//Y.data, U.data, V.data must be allocated the memory outside
	cv::Mat imgYUV;
	cv::cvtColor(*imgBGR, imgYUV, cv::COLOR_BGR2YCrCb);  //COLOR_BGR2YCrCb or COLOR_BGR2YUV
	unsigned char* pixels = (unsigned char*)(imgYUV.data);

	int width = imgYUV.cols;
	int height = imgYUV.rows;
	float dummy_Y, dummy_U, dummy_V;
	for (int i = 0; i < width*height; i++)
	{
		dummy_Y = pixels[i * 3];
		dummy_U = pixels[i * 3 + 1];
		dummy_V = pixels[i * 3 + 2];
		Y->data[i] = (float)(dummy_Y) / 255.0; //float [0:1]
		U->data[i] = (float)(dummy_U) / 255.0; //float [0:1]
		V->data[i] = (float)(dummy_V) / 255.0; //float [0:1]
	}
	return;
}

void BGRIm2MatrixUV(cv::Mat *imgBGR, ConvLayer *U, ConvLayer *V)
{
	//Y.data, U.data, V.data must be allocated the memory outside
	cv::Mat imgYUV;
	cv::cvtColor(*imgBGR, imgYUV, cv::COLOR_BGR2YCrCb);  //COLOR_BGR2YCrCb or COLOR_BGR2YUV
	unsigned char* pixels = (unsigned char*)(imgYUV.data);

	int width = imgYUV.cols;
	int height = imgYUV.rows;
	float dummy_Y, dummy_U, dummy_V;
	for (int i = 0; i < width*height; i++)
	{
		//dummy_Y = pixels[i * 3];
		dummy_U = pixels[i * 3 + 1];
		dummy_V = pixels[i * 3 + 2];
		//Y->data[i] = (float)(dummy_Y) / 255.0; //float [0:1]
		U->data[i] = (float)(dummy_U) / 255.0; //float [0:1]
		V->data[i] = (float)(dummy_V) / 255.0; //float [0:1]
	}
	return;
}


//BGR color
void Matrix2ImBGR(ConvLayer *Y, ConvLayer *U, ConvLayer *V, cv::Mat *imgBGR)
{

	cv::Mat Img(Y->height, Y->width, CV_8UC3);	unsigned char *image_ptr = Img.data;	float dummy_Y, dummy_U, dummy_V;	//std::memcpy(Img.data, c, 478 * 478 * sizeof(unsigned char));
	for (int i = 0; i < Y->width*Y->height; i++)
	{
		dummy_Y = Y->data[i] * 255.0; //float [0,255]
		if (dummy_Y < 0)
		{
			dummy_Y = 0;
		}
		else if (dummy_Y > 255)
		{
			dummy_Y = 255;
		}
		image_ptr[i * 3] = (unsigned char)dummy_Y;

		dummy_U = U->data[i] * 255.0; //float [0,255]
		if (dummy_U < 0)
		{
			dummy_U = 0;
		}
		else if (dummy_U > 255)
		{
			dummy_U = 255;
		}
		image_ptr[i * 3 + 1] = (unsigned char)dummy_U;

		dummy_V = V->data[i] * 255.0; //float [0,255]
		if (dummy_V < 0)
		{
			dummy_V = 0;
		}
		else if (dummy_V > 255)
		{
			dummy_V = 255;
		}
		image_ptr[i * 3 + 2] = (unsigned char)dummy_V;
	}

	cv::cvtColor(Img, *imgBGR, cv::COLOR_YCrCb2BGR);  //COLOR_BGR2YCrCb or COLOR_BGR2YUV
	return;

}


/***************************************************/

// Read from memory, write to handle
inline void read_from_dnnl_memory(void *handle, mkldnn::memory &mem) {
	mkldnn::engine eng = mem.get_engine();
	size_t bytes = mem.get_desc().get_size();

	if (eng.get_kind() == mkldnn::engine::kind::cpu) {
		uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
		std::copy(src, src + bytes, (uint8_t *)handle);
	}
	else
	{
		cout << "read_from_dnnl_memory error" << endl;
	}
}

// Read from handle, write to memory
inline void write_to_dnnl_memory(void *handle, mkldnn::memory &mem) {
	mkldnn::engine eng = mem.get_engine();
	size_t bytes = mem.get_desc().get_size();

	if (eng.get_kind() == mkldnn::engine::kind::cpu) {
		uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
		std::copy((uint8_t *)handle, (uint8_t *)handle + bytes, dst);
	}
	else
	{
		cout << "write_to_dnnl_memory error" << endl;
	}
}

void DRRN(int up_scale)
{
	DRRN_Model sr;

	clock_t start_t, end_t;//计算时间
	start_t = clock();

	// 加载 CNN 模型参数
	loadModel(&sr);

	end_t = clock();

	double total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	cout << "load model cost time: " << total_t << endl;


	//双三次放大
	//ResizeGrayscaleImage(&jpg, up_scale);
	cv::resize(jpg, jpg, cv::Size(jpg.cols * up_scale, jpg.rows * up_scale), 0, 0, cv::INTER_CUBIC);
	imshow("bicubic_scale", jpg);

	int wid = jpg.cols;
	int hei = jpg.rows;

	cout << "Input Image Width:" << wid << endl;
	cout << "        Height:" << hei << endl;


	


	ConvLayer im_b(wid, hei);//即Y通道
	ConvLayer U(wid, hei), V(wid, hei);
	im_b.data = new float[wid * hei];
	U.data = new float[wid * hei];
	V.data = new float[wid * hei];

	//RGB转换为YUV
	//RGB2YUV(&jpg, &im_b, &U, &V);
	BGRIm2Matrix(&jpg, &im_b, &U, &V);

	//cv::Mat im_tt; 
	//im_tt = Matrix2Im(&im_b);
	//cv::imshow("im_b", im_tt);

#if 0
	//首单位层结果
	ConvLayer convfea1(wid, hei, 128);
	convfea1.data = new float[wid * hei * 128];




	//第一部分 首单位层

	ConvLayer im_b1(wid, hei);//即Y通道
	im_b1.data = new float[wid * hei];
	copy_matrix(&im_b, &im_b1);

	cout << "the first layer... " << endl;

	vl_BatchNorm(&im_b1, sr.the_first_unit.u, sr.the_first_unit.std);//函数

											   //save_卷积层2jpg(&im_b1,"cc1");




	vl_Scale(&im_b1, sr.the_first_unit.alpha, sr.the_first_unit.beta);//函数  


	vl_nnrelu(&im_b1);//激励函数

	//float *c_bn_output = im_b1.data;
	//std::cout << "C Batch Normal" << std::endl;
	//// [Check the results]
	//for (int n = 0; n < 1; n++)
	//{
	//	for (int c = 0; c < 1; c++) {
	//		for (int h = 0; h < 1; h++)
	//		{
	//			for (int w = 0; w < wid; w++)
	//			{
	//				std::cout << std::setfill(' ') << std::setw(5) << std::setprecision(6) << (*c_bn_output);
	//				std::cout << " ";
	//				c_bn_output++;
	//			}
	//		}
	//	}
	//}
	//std::cout << std::endl;

	LayerData * CurrentLayer = sr.the_first_unit.conv_layer;

	vl_nnconv(&im_b1, &convfea1, CurrentLayer, 1, 1, 1, 1, 1, 1);

	delete[]im_b1.data;  im_b1.data = NULL;



	//第二部分  9残差块
	ConvLayer convfea2(wid, hei, 128);
	convfea2.data = new float[wid * hei * 128];
	ConvLayer convfea3(wid, hei, 128);
	convfea3.data = new float[wid * hei * 128];

	ConvLayer *Source, *Target;

	Source = &convfea2;
	Target = &convfea3;
	copy_matrix(&convfea1, Source);

	//float *c_bn_output = convfea1.data;
	//std::cout << "Orignal - The 1st Unit Conv" << std::endl;
	//// [Check the results]
	//for (int n = 0; n < wid * 2; n++)
	//{
	//	std::cout << "i: " << n << " " << std::setw(5) << std::setprecision(6) << *c_bn_output;
	//	std::cout << "  ";
	//	c_bn_output++;
	//}
	//std::cout << std::endl;
	//std::cout << std::endl;

	residual_block * residual_block0 = sr.block;

	cout << "9 residual blocks... include 2 group(BN,Scale,Convolution)" << endl;

	for (int k = 0; k<sr.residual_block_number; k++)
	{

		cout << k << endl;

		for (int p = 0; p<2; p++)
		{

			vl_BatchNorm(Source, residual_block0->data[p].u, residual_block0->data[p].std);//函数

			vl_Scale(Source, residual_block0->data[p].alpha, residual_block0->data[p].beta);//函数  

			vl_nnrelu(Source);

			if(0)
			{
				float *c_bn_output = Source->data + wid*hei * 5 + wid * 5 + 5;
				std::cout << "The C residual unit BN" << std::endl;
				// +wid*hei + wid * 5 + 5;
				// [Check the results]
				for (int n = 0; n < 32 ; n++)
				{
					std::cout << "i: " << n << " " << std::setw(5) << std::setprecision(6) << *c_bn_output;
					std::cout << "  ";
					c_bn_output++;
				}
				cout << endl;
			}

			CurrentLayer = residual_block0->data[p].conv_layer;

			vl_nnconv(Source, Target, CurrentLayer, 1, 1, 1, 1, 1, 1);

			std::swap(Source, Target);


			if (0)
			{
				float *c_bn_output = Source->data + wid*hei * 5 + wid * 5 + 5;
				std::cout << "The C residual unit Conv " << std::endl;
				// +wid*hei + wid * 5 + 5;
				// [Check the results]
				for (int n = 0; n < 32; n++)
				{
					std::cout << "i: " << n << " " << std::setw(5) << std::setprecision(6) << *c_bn_output;
					std::cout << "  ";
					c_bn_output++;
				}
				cout << endl;
			}

		}
		//求和
		add_matrix(&convfea1, Source);//和首层结果相加
									  //char txt[255];
									  //sprintf(txt, "cd%d", k);  
									  //save_卷积层2jpg(源,txt);

		if (1)
		{
			float *c_bn_output = Source->data + wid*hei * 5 + wid * 5 + 5;
			std::cout << "The C residual unit Conv + sum: "<<k << std::endl;
			// +wid*hei + wid * 5 + 5;
			// [Check the results]
			for (int n = 0; n < 32 ; n++)
			{
				std::cout << "i: " << n << " " << std::setw(5) << std::setprecision(6) << *c_bn_output;
				std::cout << "  ";
				c_bn_output++;
			}
			cout << endl;
		}
		residual_block0++;//到下残差块
	}//end


	 //尾单位层  ---->开始
	cout << "The last layer... " << endl;

	vl_BatchNorm(Source, sr.the_last_unit.u, sr.the_last_unit.std);//函数

	vl_Scale(Source, sr.the_last_unit.alpha, sr.the_last_unit.beta);//函数  

	vl_nnrelu(Source);

	if (1)
	{
		float *c_bn_output = Source->data ;
		std::cout << "The C last unit BN: " << std::endl;
		// +wid*hei + wid * 5 + 5;
		// [Check the results]
		for (int n = 0; n < 32; n++)
		{
			std::cout << "i: " << n << " " << std::setw(5) << std::setprecision(6) << *c_bn_output;
			std::cout << "  ";
			c_bn_output++;
		}
		cout << endl;
	}
	//save_mat ("dd2.txt",源->data,源->width,源->height,源->depth) ; //保存
	delete[]convfea1.data;  convfea1.data = NULL;
	delete[]Target->data;  Target->data = NULL;

	CurrentLayer = sr.the_last_unit.conv_layer;

	// 3倍重建图
	ConvLayer hR1(wid, hei);

	hR1.data = new float[wid * hei];

	vl_nnconv(Source, &hR1, CurrentLayer, 1, 1, 1, 1, 1, 1);

	if (1)
	{
		float *c_bn_output = hR1.data;
		std::cout << "The C last unit Conv: " << std::endl;
		// +wid*hei + wid * 5 + 5;
		// [Check the results]
		for (int n = 0; n < 32; n++)
		{
			std::cout << "i: " << n << " " << std::setw(5) << std::setprecision(6) << *c_bn_output;
			std::cout << "  ";
			c_bn_output++;
		}
		cout << endl;
	}
	//尾单位层  <----结束
	delete[]Source->data;  Source->data = NULL;


	//求和
	cout << "Combine output image... " << endl;

	add_matrix(&im_b, &hR1);
	if (1)
	{
		float *c_bn_output = hR1.data;
		std::cout << "The C last output: " << std::endl;
		// +wid*hei + wid * 5 + 5;
		// [Check the results]
		for (int n = 0; n < 32; n++)
		{
			std::cout << "i: " << n << " " << std::setw(5) << std::setprecision(6) << *c_bn_output;
			std::cout << "  ";
			c_bn_output++;
		}
		cout << endl;
	}
	//cv::Mat im_tt2;
	//im_tt2 = Matrix2Im(&hR1);
	//cv::imshow("gray", im_tt2);



#endif

	/**********************************************************/
	engine cpu_engine(engine::kind::cpu, 0);
	stream cpu_stream(cpu_engine);

	//第一部分，第一层bn+relu+conv
	int N = 1, H = hei, W = wid, C = 1;
	int first_unit_bn_mean_size = C;
	int first_unit_bn_scale_shift_size = 2 * C;
	int image_size = N * H * W * C;

	std::vector<float> image(image_size);
	std::vector<float> first_unit_bn_mean(first_unit_bn_mean_size);
	std::vector<float> first_unit_bn_var(first_unit_bn_mean_size);
	std::vector<float> first_unit_bn_scale_shift(first_unit_bn_scale_shift_size);

	for (int offset = 0; offset < N*C*H*W; offset++)
	{
		image[offset] = im_b.data[offset];
	}
	
	for (int n = 0; n < first_unit_bn_mean_size; n++)
	{
		first_unit_bn_mean[n] =  sr.the_first_unit.u[n];
		first_unit_bn_var[n] =  sr.the_first_unit.std[n];
	}
	for (int n = 0; n < first_unit_bn_scale_shift_size/2; n++)
	{
		first_unit_bn_scale_shift[n] = sr.the_first_unit.alpha[n];    //scale
		first_unit_bn_scale_shift[n + first_unit_bn_scale_shift_size / 2] = sr.the_first_unit.beta[n];  //shift
	}

	memory::dims first_unit_mean_tz = { C };
	memory::dims first_unit_scale_shift_tz = { 2, C };
	memory::dims first_unit_src_tz = { N, C, H, W };

	auto first_unit_bn_mean_md = memory::desc(first_unit_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto first_unit_bn_scale_shift_md = memory::desc(first_unit_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto user_src_md = memory::desc(
		first_unit_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto user_src_mem = memory(user_src_md, cpu_engine); 
	write_to_dnnl_memory(image.data(), user_src_mem);
	auto first_unit_mean_mem = memory(first_unit_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(first_unit_bn_mean.data(), first_unit_mean_mem);
	auto first_unit_var_mem = memory(first_unit_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(first_unit_bn_var.data(), first_unit_var_mem);
	auto first_unit_scale_shift_mem = memory(first_unit_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(first_unit_bn_scale_shift.data(), first_unit_scale_shift_mem);
	auto first_layer_bn_dst_mem = memory(user_src_md, cpu_engine);

	normalization_flags flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto first_unit_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		user_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);


	mkldnn::post_ops po1;
	//po.append_sum(
	//	/* scale = */ 1.f);
	po1.append_eltwise(
		/* scale     = */ 1.f,
		/* alg kind  = */ mkldnn::algorithm::eltwise_relu,
		/* neg slope = */ 0.f,
		/* unused for relu */ 0.f);
	mkldnn::primitive_attr attr1;
	attr1.set_post_ops(po1);

	//auto first_unit_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(first_unit_bnrm_fwd_d, cpu_engine);
	auto first_unit_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(first_unit_bnrm_fwd_d, attr1, cpu_engine);
	auto first_unit_bnrm_fwd = batch_normalization_forward(first_unit_bnrm_fwd_pd);
	first_unit_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, user_src_mem },
			{ MKLDNN_ARG_MEAN, first_unit_mean_mem },
			{ MKLDNN_ARG_VARIANCE, first_unit_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, first_unit_scale_shift_mem },
			{ MKLDNN_ARG_DST, first_layer_bn_dst_mem }
		}
	);

	//float *conv3_bn_output = static_cast<float *>(first_layer_bn_dst_mem.get_data_handle());
	//std::cout << "Batch Normal" << std::endl;
	//// [Check the results]
	//for (int n = 0; n < 1; n++)
	//{
	//	for (int c = 0; c < 1; c++) {
	//		for (int h = 0; h < 1; h++)
	//		{
	//			for (int w = 0; w < W; w++)
	//			{
	//				std::cout << std::setfill(' ') << std::setw(5) << std::setprecision(6) << (*conv3_bn_output);
	//				std::cout << " ";
	//				conv3_bn_output++;
	//			}
	//		}
	//	}
	//}
	//std::cout << std::endl;

	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 1;
	int IC = C, OC = 128, KH = 3, KW = 3;

	int first_unit_weights_size = OC* IC * KW * KH;
	int first_unit_bias_size = OC;
	std::vector<float> first_unit_weights(first_unit_weights_size);
	std::vector<float> first_unit_bias(first_unit_bias_size);


	for (int offset = 0; offset < first_unit_weights_size; offset++)
	{
		first_unit_weights[offset] = sr.the_first_unit.conv_layer->weights_data[offset];
	}

	for (int offset = 0; offset < first_unit_bias_size; offset++)
	{
		first_unit_bias[offset] = sr.the_first_unit.conv_layer->bias_data[offset];
	}

	memory::dims first_unit_conv_src_tz = { N, C, H, W };
	memory::dims first_unit_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims first_unit_conv_bias_tz = { OC };
	memory::dims first_unit_conv_dst_tz = { N, OC, H, W };
	memory::dims conv_strides = { 1, 1 };
	memory::dims conv_padding = { 1, 1 };



	auto first_unit_conv_src_md = memory::desc({ first_unit_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto first_unit_conv_bias_md = memory::desc({ first_unit_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto first_unit_conv_weights_md = memory::desc({ first_unit_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto first_unit_conv_dst_md = memory::desc({ first_unit_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto user_conv1_weights_md = memory::desc(
		first_unit_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto user_conv1_bias_md = memory::desc({ first_unit_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	auto user_conv1_weights_mem = memory(user_conv1_weights_md, cpu_engine);
	write_to_dnnl_memory(first_unit_weights.data(), user_conv1_weights_mem);
	auto user_conv1_bias_mem = memory(user_conv1_bias_md, cpu_engine);
	write_to_dnnl_memory(first_unit_bias.data(), user_conv1_bias_mem);

	//[Create convolution descriptor]
	auto first_unit_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, first_unit_conv_src_md, first_unit_conv_weights_md,
		first_unit_conv_bias_md, first_unit_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto first_unit_conv1_fast_prim_desc = convolution_forward::primitive_desc(first_unit_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
	auto first_unit_conv1_src_memory = first_layer_bn_dst_mem;
	if (first_unit_conv1_fast_prim_desc.src_desc() != first_layer_bn_dst_mem.get_desc()) {
		first_unit_conv1_src_memory = memory(first_unit_conv1_fast_prim_desc.src_desc(), cpu_engine);
		reorder(first_layer_bn_dst_mem, first_unit_conv1_src_memory)
			.execute(cpu_stream, first_layer_bn_dst_mem, first_unit_conv1_src_memory);
	}
	auto first_unit_conv1_weights_memory = user_conv1_weights_mem;
	if (first_unit_conv1_fast_prim_desc.weights_desc() != user_conv1_weights_mem.get_desc()) {
		first_unit_conv1_weights_memory = memory(first_unit_conv1_fast_prim_desc.weights_desc(), cpu_engine);
		reorder(user_conv1_weights_mem, first_unit_conv1_weights_memory)
			.execute(cpu_stream, user_conv1_weights_mem, first_unit_conv1_weights_memory);
	}

	//[Create memory for output]
	auto first_unit_conv1_dst_memory = memory(first_unit_conv1_fast_prim_desc.dst_desc(), cpu_engine);
	//[Create memory for output]
	// create convolution primitive and add it to net
	auto first_unit_fast_conv1 = convolution_forward(first_unit_conv1_fast_prim_desc);

	first_unit_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, first_unit_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS, first_unit_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, user_conv1_bias_mem },
			{ MKLDNN_ARG_DST, first_unit_conv1_dst_memory }
		}
	);

	auto first_unit_user_dst_md = memory::desc(
		first_unit_conv_dst_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case
	);
	auto first_unit_user_conv1_dst_mem = memory(first_unit_user_dst_md, cpu_engine);  //for conv output

																  // create reorder between internal and user data if it is needed and
																  // add it to net after pooling
	if (first_unit_conv1_dst_memory != first_unit_user_conv1_dst_mem) {
		reorder(first_unit_conv1_dst_memory, first_unit_user_conv1_dst_mem)
			.execute(cpu_stream, first_unit_conv1_dst_memory, first_unit_user_conv1_dst_mem);
	}

	cpu_stream.wait();
	int first_unit_conv_output_size = N*OC*H*W;
	std::vector<float> first_unit_conv_output_mem(first_unit_conv_output_size);
	read_from_dnnl_memory(first_unit_conv_output_mem.data(), first_unit_user_conv1_dst_mem);

	//{
	//	float * conv3_conv_output = static_cast<float *>(first_unit_user_conv1_dst_mem.get_data_handle());
	//	std::cout << "The first unit 1 Conv" << std::endl;
	//	conv3_conv_output = conv3_conv_output + wid*hei * 5 + wid * 5 + 5;
	//	// [Check the results]
	//	for (int n = 0; n < 32; n++)
	//	{
	//		std::cout << "i: " << n << " " << std::setw(5) << std::setprecision(6) << *conv3_conv_output;
	//		std::cout << "  ";
	//		conv3_conv_output++;
	//	}
	//	std::cout << std::endl;
	//}


	/************* the first layer ends here*********************************************/
	//第二部分  9残差块
#if 1
	/*the 1-0 residual block*/
	/**********************BN***********************************/
	N = 1, H = hei, W = wid, C = 128;
	int residual_unit1_0_bn_mean_size = C;
	int residual_unit1_0_bn_scale_shift_size = 2 * C;

	std::vector<float> residual_unit1_0_bn_mean(residual_unit1_0_bn_mean_size);
	std::vector<float> residual_unit1_0_bn_var(residual_unit1_0_bn_mean_size);
	std::vector<float> residual_unit1_0_bn_scale_shift(residual_unit1_0_bn_scale_shift_size);

	residual_block * residual_block0 = sr.block;

	for (int n = 0; n < residual_unit1_0_bn_mean_size; n++)
	{
		residual_unit1_0_bn_mean[n] = residual_block0->data[0].u[n];
		residual_unit1_0_bn_var[n] =  residual_block0->data[0].std[n];
	}
	for (int n = 0; n < residual_unit1_0_bn_scale_shift_size / 2; n++)
	{
		residual_unit1_0_bn_scale_shift[n] =  residual_block0->data[0].alpha[n];    //scale
		residual_unit1_0_bn_scale_shift[n + residual_unit1_0_bn_scale_shift_size / 2] =  residual_block0->data[0].beta[n];  //shift
	}

	memory::dims residual_unit1_0_mean_tz = { C };
	memory::dims residual_unit1_0_scale_shift_tz = { 2, C };
	memory::dims residual_unit1_0_src_tz = { N, C, H, W };

	auto residual_unit1_0_bn_mean_md = memory::desc(residual_unit1_0_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto residual_unit1_0_bn_scale_shift_md = memory::desc(residual_unit1_0_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto residual_unit1_0_src_md = memory::desc(
		residual_unit1_0_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto residual_unit1_0_src_mem = first_unit_user_conv1_dst_mem; // 
	auto residual_unit1_0_mean_mem = memory(residual_unit1_0_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit1_0_bn_mean.data(), residual_unit1_0_mean_mem);
	auto residual_unit1_0_var_mem = memory(residual_unit1_0_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit1_0_bn_var.data(), residual_unit1_0_var_mem);
	auto residual_unit1_0_scale_shift_mem = memory(residual_unit1_0_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(residual_unit1_0_bn_scale_shift.data(), residual_unit1_0_scale_shift_mem);
	auto residual_unit1_0_bn_dst_mem = memory(residual_unit1_0_src_md, cpu_engine);

	flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto residual_unit1_0_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		residual_unit1_0_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);

	auto residual_unit1_0_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(residual_unit1_0_bnrm_fwd_d, attr1, cpu_engine);
	auto residual_unit1_0_bnrm_fwd = batch_normalization_forward(residual_unit1_0_bnrm_fwd_pd);
	residual_unit1_0_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, residual_unit1_0_src_mem },
			{ MKLDNN_ARG_MEAN, residual_unit1_0_mean_mem },
			{ MKLDNN_ARG_VARIANCE, residual_unit1_0_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, residual_unit1_0_scale_shift_mem },
			{ MKLDNN_ARG_DST, residual_unit1_0_bn_dst_mem }
		}
	);

	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 128;
	IC = C, OC = 128, KH = 3, KW = 3;

	int  residual_unit1_0_weights_size = OC* IC * KW * KH;
	int  residual_unit1_0_bias_size = OC;
	std::vector<float>  residual_unit1_0_weights( residual_unit1_0_weights_size);
	std::vector<float>  residual_unit1_0_bias( residual_unit1_0_bias_size);


	for (int offset = 0; offset <  residual_unit1_0_weights_size; offset++)
	{
		 residual_unit1_0_weights[offset] = residual_block0->data[0].conv_layer->weights_data[offset];
	}

	for (int offset = 0; offset <  residual_unit1_0_bias_size; offset++)
	{
		 residual_unit1_0_bias[offset] = residual_block0->data[0].conv_layer->bias_data[offset];
	}

	memory::dims  residual_unit1_0_conv_src_tz = { N, C, H, W };
	memory::dims  residual_unit1_0_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims  residual_unit1_0_conv_bias_tz = { OC };
	memory::dims  residual_unit1_0_conv_dst_tz = { N, OC, H, W };


	auto  residual_unit1_0_conv_src_md = memory::desc({  residual_unit1_0_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit1_0_conv_bias_md = memory::desc({  residual_unit1_0_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit1_0_conv_weights_md = memory::desc({  residual_unit1_0_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit1_0_conv_dst_md = memory::desc({  residual_unit1_0_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto residual_unit1_0_user_conv1_weights_md = memory::desc(
		 residual_unit1_0_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto residual_unit1_0_user_conv1_bias_md = memory::desc({  residual_unit1_0_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	auto residual_unit1_0_user_conv1_weights_mem = memory(residual_unit1_0_user_conv1_weights_md, cpu_engine);
	write_to_dnnl_memory( residual_unit1_0_weights.data(), residual_unit1_0_user_conv1_weights_mem);
	auto residual_unit1_0_user_conv1_bias_mem = memory(residual_unit1_0_user_conv1_bias_md, cpu_engine);
	write_to_dnnl_memory( residual_unit1_0_bias.data(), residual_unit1_0_user_conv1_bias_mem);

	//[Create convolution descriptor]
	auto  residual_unit1_0_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct,  residual_unit1_0_conv_src_md,  residual_unit1_0_conv_weights_md,
		 residual_unit1_0_conv_bias_md,  residual_unit1_0_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto  residual_unit1_0_conv1_fast_prim_desc = convolution_forward::primitive_desc( residual_unit1_0_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
	auto  residual_unit1_0_conv1_src_memory = residual_unit1_0_bn_dst_mem;
	if ( residual_unit1_0_conv1_fast_prim_desc.src_desc() != residual_unit1_0_bn_dst_mem.get_desc()) {
		 residual_unit1_0_conv1_src_memory = memory( residual_unit1_0_conv1_fast_prim_desc.src_desc(), cpu_engine);
		reorder(residual_unit1_0_bn_dst_mem,  residual_unit1_0_conv1_src_memory)
			.execute(cpu_stream, residual_unit1_0_bn_dst_mem,  residual_unit1_0_conv1_src_memory);
	}
	auto  residual_unit1_0_conv1_weights_memory = residual_unit1_0_user_conv1_weights_mem;
	if ( residual_unit1_0_conv1_fast_prim_desc.weights_desc() != residual_unit1_0_user_conv1_weights_mem.get_desc()) {
		 residual_unit1_0_conv1_weights_memory = memory( residual_unit1_0_conv1_fast_prim_desc.weights_desc(), cpu_engine);
		reorder(residual_unit1_0_user_conv1_weights_mem,  residual_unit1_0_conv1_weights_memory)
			.execute(cpu_stream, residual_unit1_0_user_conv1_weights_mem,  residual_unit1_0_conv1_weights_memory);
	}

	//[Create memory for output]
	auto  residual_unit1_0_conv1_dst_memory = memory( residual_unit1_0_conv1_fast_prim_desc.dst_desc(), cpu_engine);
	//[Create memory for output]
	// create convolution primitive and add it to net
	auto  residual_unit1_0_fast_conv1 = convolution_forward( residual_unit1_0_conv1_fast_prim_desc);

	 residual_unit1_0_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC,  residual_unit1_0_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS,  residual_unit1_0_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, residual_unit1_0_user_conv1_bias_mem },
			{ MKLDNN_ARG_DST,  residual_unit1_0_conv1_dst_memory }
		}
	);

	auto  residual_unit1_0_user_dst_md = memory::desc(
		 residual_unit1_0_conv_dst_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case
	);
	auto  residual_unit1_0_user_conv1_dst_mem = memory( residual_unit1_0_user_dst_md, cpu_engine);  //for conv output

																					  // create reorder between internal and user data if it is needed and
																					  // add it to net after pooling
	if ( residual_unit1_0_conv1_dst_memory !=  residual_unit1_0_user_conv1_dst_mem) {
		reorder( residual_unit1_0_conv1_dst_memory,  residual_unit1_0_user_conv1_dst_mem)
			.execute(cpu_stream,  residual_unit1_0_conv1_dst_memory,  residual_unit1_0_user_conv1_dst_mem);
	}



	/*the 1-1 residual block*/
	/**********************BN***********************************/
	N = 1, H = hei, W = wid, C = 128;
	int residual_unit1_1_bn_mean_size = C;
	int residual_unit1_1_bn_scale_shift_size = 2 * C;

	std::vector<float> residual_unit1_1_bn_mean(residual_unit1_1_bn_mean_size);
	std::vector<float> residual_unit1_1_bn_var(residual_unit1_1_bn_mean_size);
	std::vector<float> residual_unit1_1_bn_scale_shift(residual_unit1_1_bn_scale_shift_size);

	/*residual_block **/ 
	for (int n = 0; n < residual_unit1_1_bn_mean_size; n++)
	{
		residual_unit1_1_bn_mean[n] = residual_block0->data[1].u[n];
		residual_unit1_1_bn_var[n] = residual_block0->data[1].std[n];
	}
	for (int n = 0; n < residual_unit1_1_bn_scale_shift_size / 2; n++)
	{
		residual_unit1_1_bn_scale_shift[n] = residual_block0->data[1].alpha[n];    //scale
		residual_unit1_1_bn_scale_shift[n + residual_unit1_1_bn_scale_shift_size / 2] = residual_block0->data[1].beta[n];  //shift
	}

	memory::dims residual_unit1_1_mean_tz = { C };
	memory::dims residual_unit1_1_scale_shift_tz = { 2, C };
	memory::dims residual_unit1_1_src_tz = { N, C, H, W };

	auto residual_unit1_1_bn_mean_md = memory::desc(residual_unit1_1_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto residual_unit1_1_bn_scale_shift_md = memory::desc(residual_unit1_1_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto residual_unit1_1_src_md = memory::desc(
		residual_unit1_1_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto residual_unit1_1_src_mem = residual_unit1_0_user_conv1_dst_mem; // 
	auto residual_unit1_1_mean_mem = memory(residual_unit1_1_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit1_1_bn_mean.data(), residual_unit1_1_mean_mem);
	auto residual_unit1_1_var_mem = memory(residual_unit1_1_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit1_1_bn_var.data(), residual_unit1_1_var_mem);
	auto residual_unit1_1_scale_shift_mem = memory(residual_unit1_1_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(residual_unit1_1_bn_scale_shift.data(), residual_unit1_1_scale_shift_mem);
	auto residual_unit1_1_bn_dst_mem = memory(residual_unit1_1_src_md, cpu_engine);

	flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto residual_unit1_1_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		residual_unit1_1_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);

	auto residual_unit1_1_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(residual_unit1_1_bnrm_fwd_d, attr1, cpu_engine);
	auto residual_unit1_1_bnrm_fwd = batch_normalization_forward(residual_unit1_1_bnrm_fwd_pd);
	residual_unit1_1_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, residual_unit1_1_src_mem },
			{ MKLDNN_ARG_MEAN, residual_unit1_1_mean_mem },
			{ MKLDNN_ARG_VARIANCE, residual_unit1_1_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, residual_unit1_1_scale_shift_mem },
			{ MKLDNN_ARG_DST, residual_unit1_1_bn_dst_mem }
		}
	);

	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 128;
	IC = C, OC = 128, KH = 3, KW = 3;

	int  residual_unit1_1_weights_size = OC* IC * KW * KH;
	int  residual_unit1_1_bias_size = OC;
	std::vector<float>  residual_unit1_1_weights(residual_unit1_1_weights_size);
	std::vector<float>  residual_unit1_1_bias(residual_unit1_1_bias_size);


	for (int offset = 0; offset < residual_unit1_1_weights_size; offset++)
	{
		residual_unit1_1_weights[offset] = residual_block0->data[1].conv_layer->weights_data[offset];
	}

	for (int offset = 0; offset < residual_unit1_1_bias_size; offset++)
	{
		residual_unit1_1_bias[offset] = residual_block0->data[1].conv_layer->bias_data[offset];
	}

	memory::dims  residual_unit1_1_conv_src_tz = { N, C, H, W };
	memory::dims  residual_unit1_1_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims  residual_unit1_1_conv_bias_tz = { OC };
	memory::dims  residual_unit1_1_conv_dst_tz = { N, OC, H, W };


	auto  residual_unit1_1_conv_src_md = memory::desc({ residual_unit1_1_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit1_1_conv_bias_md = memory::desc({ residual_unit1_1_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit1_1_conv_weights_md = memory::desc({ residual_unit1_1_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit1_1_conv_dst_md = memory::desc({ residual_unit1_1_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto residual_unit1_1_user_conv1_weights_md = memory::desc(
		residual_unit1_1_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto residual_unit1_1_user_conv1_bias_md = memory::desc({ residual_unit1_1_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	auto residual_unit1_1_user_conv1_weights_mem = memory(residual_unit1_1_user_conv1_weights_md, cpu_engine);
	write_to_dnnl_memory(residual_unit1_1_weights.data(), residual_unit1_1_user_conv1_weights_mem);
	auto residual_unit1_1_user_conv1_bias_mem = memory(residual_unit1_1_user_conv1_bias_md, cpu_engine);
	write_to_dnnl_memory(residual_unit1_1_bias.data(), residual_unit1_1_user_conv1_bias_mem);

	//[Create convolution descriptor]
	auto  residual_unit1_1_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, residual_unit1_1_conv_src_md, residual_unit1_1_conv_weights_md,
		residual_unit1_1_conv_bias_md, residual_unit1_1_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto  residual_unit1_1_conv1_fast_prim_desc = convolution_forward::primitive_desc(residual_unit1_1_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
	auto  residual_unit1_1_conv1_src_memory = residual_unit1_1_bn_dst_mem;
	if (residual_unit1_1_conv1_fast_prim_desc.src_desc() != residual_unit1_1_bn_dst_mem.get_desc()) {
		residual_unit1_1_conv1_src_memory = memory(residual_unit1_1_conv1_fast_prim_desc.src_desc(), cpu_engine);
		reorder(residual_unit1_1_bn_dst_mem, residual_unit1_1_conv1_src_memory)
			.execute(cpu_stream, residual_unit1_1_bn_dst_mem, residual_unit1_1_conv1_src_memory);
	}
	auto  residual_unit1_1_conv1_weights_memory = residual_unit1_1_user_conv1_weights_mem;
	if (residual_unit1_1_conv1_fast_prim_desc.weights_desc() != residual_unit1_1_user_conv1_weights_mem.get_desc()) {
		residual_unit1_1_conv1_weights_memory = memory(residual_unit1_1_conv1_fast_prim_desc.weights_desc(), cpu_engine);
		reorder(residual_unit1_1_user_conv1_weights_mem, residual_unit1_1_conv1_weights_memory)
			.execute(cpu_stream, residual_unit1_1_user_conv1_weights_mem, residual_unit1_1_conv1_weights_memory);
	}

	//[Create memory for output]
	auto  residual_unit1_1_conv1_dst_memory = memory(residual_unit1_1_conv1_fast_prim_desc.dst_desc(), cpu_engine);
	//[Create memory for output]
	// create convolution primitive and add it to net
	auto  residual_unit1_1_fast_conv1 = convolution_forward(residual_unit1_1_conv1_fast_prim_desc);

	residual_unit1_1_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC,  residual_unit1_1_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS,  residual_unit1_1_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, residual_unit1_1_user_conv1_bias_mem },
			{ MKLDNN_ARG_DST,  residual_unit1_1_conv1_dst_memory }
		}
	);

#if 0
	//without sum to the first unit output
	auto  residual_unit1_1_user_dst_md = memory::desc(
		residual_unit1_1_conv_dst_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case
	);
	auto  residual_unit1_1_user_conv1_dst_mem = memory(residual_unit1_1_user_dst_md, cpu_engine);  //for conv output

																								   // create reorder between internal and user data if it is needed and

	
	// add it to net after pooling
	if (residual_unit1_1_conv1_dst_memory != residual_unit1_1_user_conv1_dst_mem) {
		reorder(residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem)
			.execute(cpu_stream, residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem);
	
#else

	//with sum to the first unit output
	mkldnn::post_ops po2;
	po2.append_sum(1.0f);
	primitive_attr attr2;
	attr2.set_post_ops(po2);

	auto  residual_unit1_1_user_conv1_dst_mem = memory(first_unit_user_dst_md, cpu_engine);
	write_to_dnnl_memory(first_unit_conv_output_mem.data(), residual_unit1_1_user_conv1_dst_mem);

	auto residual_unit1_src_reorder_pd = reorder::primitive_desc(residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem,attr2);
	reorder(residual_unit1_src_reorder_pd).execute(cpu_stream, residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem);
#endif

	/*the 2-0 residual block*/
	/**********************BN***********************************/
	N = 1, H = hei, W = wid, C = 128;
	int residual_unit2_0_bn_mean_size = C;
	int residual_unit2_0_bn_scale_shift_size = 2 * C;

	std::vector<float> residual_unit2_0_bn_mean(residual_unit2_0_bn_mean_size);
	std::vector<float> residual_unit2_0_bn_var(residual_unit2_0_bn_mean_size);
	std::vector<float> residual_unit2_0_bn_scale_shift(residual_unit2_0_bn_scale_shift_size);

	/*residual_block **/ residual_block0++;
	for (int n = 0; n < residual_unit2_0_bn_mean_size; n++)
	{
		residual_unit2_0_bn_mean[n] = residual_block0->data[0].u[n];
		residual_unit2_0_bn_var[n] = residual_block0->data[0].std[n];
	}
	for (int n = 0; n < residual_unit2_0_bn_scale_shift_size / 2; n++)
	{
		residual_unit2_0_bn_scale_shift[n] = residual_block0->data[0].alpha[n];    //scale
		residual_unit2_0_bn_scale_shift[n + residual_unit2_0_bn_scale_shift_size / 2] = residual_block0->data[0].beta[n];  //shift
	}

	memory::dims residual_unit2_0_mean_tz = { C };
	memory::dims residual_unit2_0_scale_shift_tz = { 2, C };
	memory::dims residual_unit2_0_src_tz = { N, C, H, W };

	auto residual_unit2_0_bn_mean_md = memory::desc(residual_unit2_0_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto residual_unit2_0_bn_scale_shift_md = memory::desc(residual_unit2_0_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto residual_unit2_0_src_md = memory::desc(
		residual_unit2_0_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto residual_unit2_0_src_mem = residual_unit1_1_user_conv1_dst_mem; // 
	auto residual_unit2_0_mean_mem = memory(residual_unit2_0_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit2_0_bn_mean.data(), residual_unit2_0_mean_mem);
	auto residual_unit2_0_var_mem = memory(residual_unit2_0_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit2_0_bn_var.data(), residual_unit2_0_var_mem);
	auto residual_unit2_0_scale_shift_mem = memory(residual_unit2_0_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(residual_unit2_0_bn_scale_shift.data(), residual_unit2_0_scale_shift_mem);

	flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto residual_unit2_0_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		residual_unit2_0_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);

	auto residual_unit2_0_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(residual_unit2_0_bnrm_fwd_d, attr1, cpu_engine);
	auto residual_unit2_0_bnrm_fwd = batch_normalization_forward(residual_unit2_0_bnrm_fwd_pd);
	residual_unit2_0_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, residual_unit2_0_src_mem },
			{ MKLDNN_ARG_MEAN, residual_unit2_0_mean_mem },
			{ MKLDNN_ARG_VARIANCE, residual_unit2_0_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, residual_unit2_0_scale_shift_mem },
			{ MKLDNN_ARG_DST, residual_unit1_0_bn_dst_mem }
		}
	);

	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 128;
	IC = C, OC = 128, KH = 3, KW = 3;

	memory::dims  residual_unit2_0_conv_src_tz = { N, C, H, W };
	memory::dims  residual_unit2_0_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims  residual_unit2_0_conv_bias_tz = { OC };
	memory::dims  residual_unit2_0_conv_dst_tz = { N, OC, H, W };


	auto  residual_unit2_0_conv_src_md = memory::desc({ residual_unit2_0_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit2_0_conv_bias_md = memory::desc({ residual_unit2_0_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit2_0_conv_weights_md = memory::desc({ residual_unit2_0_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit2_0_conv_dst_md = memory::desc({ residual_unit2_0_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto residual_unit2_0_user_conv1_weights_md = memory::desc(
		residual_unit2_0_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto residual_unit2_0_user_conv1_bias_md = memory::desc({ residual_unit2_0_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	//[Create convolution descriptor]
	auto  residual_unit2_0_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, residual_unit2_0_conv_src_md, residual_unit2_0_conv_weights_md,
		residual_unit2_0_conv_bias_md, residual_unit2_0_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto  residual_unit2_0_conv1_fast_prim_desc = convolution_forward::primitive_desc(residual_unit2_0_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
		reorder(residual_unit1_0_bn_dst_mem, residual_unit1_0_conv1_src_memory)
			.execute(cpu_stream, residual_unit1_0_bn_dst_mem, residual_unit1_0_conv1_src_memory);

	//[Create memory for output]
	// create convolution primitive and add it to net
	auto  residual_unit2_0_fast_conv1 = convolution_forward(residual_unit2_0_conv1_fast_prim_desc);

	residual_unit2_0_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC,  residual_unit1_0_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS,  residual_unit1_0_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, residual_unit1_0_user_conv1_bias_mem },
			{ MKLDNN_ARG_DST,  residual_unit1_0_conv1_dst_memory }
		}
	);

																								   // create reorder between internal and user data if it is needed and
																								   // add it to net after pooling
	if (residual_unit1_0_conv1_dst_memory != residual_unit1_0_user_conv1_dst_mem) {
		reorder(residual_unit1_0_conv1_dst_memory, residual_unit1_0_user_conv1_dst_mem)
			.execute(cpu_stream, residual_unit1_0_conv1_dst_memory, residual_unit1_0_user_conv1_dst_mem);
	}


	/*the 2-1 residual block*/
	/**********************BN***********************************/
	N = 1, H = hei, W = wid, C = 128;
	int residual_unit2_1_bn_mean_size = C;
	int residual_unit2_1_bn_scale_shift_size = 2 * C;

	std::vector<float> residual_unit2_1_bn_mean(residual_unit2_1_bn_mean_size);
	std::vector<float> residual_unit2_1_bn_var(residual_unit2_1_bn_mean_size);
	std::vector<float> residual_unit2_1_bn_scale_shift(residual_unit2_1_bn_scale_shift_size);

	/*residual_block **/ 
	for (int n = 0; n < residual_unit2_1_bn_mean_size; n++)
	{
		residual_unit2_1_bn_mean[n] =  residual_block0->data[1].u[n];
		residual_unit2_1_bn_var[n] =  residual_block0->data[1].std[n];
	}
	for (int n = 0; n < residual_unit2_1_bn_scale_shift_size / 2; n++)
	{
		residual_unit2_1_bn_scale_shift[n] = residual_block0->data[1].alpha[n];    //scale
		residual_unit2_1_bn_scale_shift[n + residual_unit2_1_bn_scale_shift_size / 2] = residual_block0->data[1].beta[n];  //shift
	}

	memory::dims residual_unit2_1_mean_tz = { C };
	memory::dims residual_unit2_1_scale_shift_tz = { 2, C };
	memory::dims residual_unit2_1_src_tz = { N, C, H, W };

	auto residual_unit2_1_bn_mean_md = memory::desc(residual_unit2_1_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto residual_unit2_1_bn_scale_shift_md = memory::desc(residual_unit2_1_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto residual_unit2_1_src_md = memory::desc(
		residual_unit2_1_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto residual_unit2_1_src_mem = residual_unit1_0_user_conv1_dst_mem; // 
	auto residual_unit2_1_mean_mem = memory(residual_unit2_1_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit2_1_bn_mean.data(), residual_unit2_1_mean_mem);
	auto residual_unit2_1_var_mem = memory(residual_unit2_1_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit2_1_bn_var.data(), residual_unit2_1_var_mem);
	auto residual_unit2_1_scale_shift_mem = memory(residual_unit2_1_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(residual_unit2_1_bn_scale_shift.data(), residual_unit2_1_scale_shift_mem);

	flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto residual_unit2_1_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		residual_unit2_1_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);

	auto residual_unit2_1_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(residual_unit2_1_bnrm_fwd_d, attr1, cpu_engine);
	auto residual_unit2_1_bnrm_fwd = batch_normalization_forward(residual_unit2_1_bnrm_fwd_pd);
	residual_unit2_1_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, residual_unit2_1_src_mem },
			{ MKLDNN_ARG_MEAN, residual_unit2_1_mean_mem },
			{ MKLDNN_ARG_VARIANCE, residual_unit2_1_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, residual_unit2_1_scale_shift_mem },
			{ MKLDNN_ARG_DST, residual_unit1_1_bn_dst_mem }
		}
	);

	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 128;
	IC = C, OC = 128, KH = 3, KW = 3;

	memory::dims  residual_unit2_1_conv_src_tz = { N, C, H, W };
	memory::dims  residual_unit2_1_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims  residual_unit2_1_conv_bias_tz = { OC };
	memory::dims  residual_unit2_1_conv_dst_tz = { N, OC, H, W };


	auto  residual_unit2_1_conv_src_md = memory::desc({ residual_unit2_1_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit2_1_conv_bias_md = memory::desc({ residual_unit2_1_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit2_1_conv_weights_md = memory::desc({ residual_unit2_1_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit2_1_conv_dst_md = memory::desc({ residual_unit2_1_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto residual_unit2_1_user_conv1_weights_md = memory::desc(
		residual_unit2_1_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto residual_unit2_1_user_conv1_bias_md = memory::desc({ residual_unit2_1_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	//[Create convolution descriptor]
	auto  residual_unit2_1_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, residual_unit2_1_conv_src_md, residual_unit2_1_conv_weights_md,
		residual_unit2_1_conv_bias_md, residual_unit2_1_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto  residual_unit2_1_conv1_fast_prim_desc = convolution_forward::primitive_desc(residual_unit2_1_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
		reorder(residual_unit1_1_bn_dst_mem, residual_unit1_1_conv1_src_memory)
			.execute(cpu_stream, residual_unit1_1_bn_dst_mem, residual_unit1_1_conv1_src_memory);


	//[Create memory for output]
	// create convolution primitive and add it to net
	auto  residual_unit2_1_fast_conv1 = convolution_forward(residual_unit2_1_conv1_fast_prim_desc);

	residual_unit2_1_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC,  residual_unit1_1_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS,  residual_unit1_1_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, residual_unit1_1_user_conv1_bias_mem },
			{ MKLDNN_ARG_DST,  residual_unit1_1_conv1_dst_memory }
		}
	);

	//with sum to the first unit output
	write_to_dnnl_memory(first_unit_conv_output_mem.data(), residual_unit1_1_user_conv1_dst_mem);

	auto residual_unit2_src_reorder_pd = reorder::primitive_desc(residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem, attr2);
	reorder(residual_unit2_src_reorder_pd).execute(cpu_stream, residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem);


	/*the 3-0 residual block*/
	/**********************BN***********************************/
	N = 1, H = hei, W = wid, C = 128;
	int residual_unit3_0_bn_mean_size = C;
	int residual_unit3_0_bn_scale_shift_size = 2 * C;

	std::vector<float> residual_unit3_0_bn_mean(residual_unit3_0_bn_mean_size);
	std::vector<float> residual_unit3_0_bn_var(residual_unit3_0_bn_mean_size);
	std::vector<float> residual_unit3_0_bn_scale_shift(residual_unit3_0_bn_scale_shift_size);

	/*residual_block **/ residual_block0++;
	for (int n = 0; n < residual_unit3_0_bn_mean_size; n++)
	{
		residual_unit3_0_bn_mean[n] = residual_block0->data[0].u[n];
		residual_unit3_0_bn_var[n] = residual_block0->data[0].std[n];
	}
	for (int n = 0; n < residual_unit3_0_bn_scale_shift_size / 2; n++)
	{
		residual_unit3_0_bn_scale_shift[n] = residual_block0->data[0].alpha[n];    //scale
		residual_unit3_0_bn_scale_shift[n + residual_unit3_0_bn_scale_shift_size / 2] = residual_block0->data[0].beta[n];  //shift
	}

	memory::dims residual_unit3_0_mean_tz = { C };
	memory::dims residual_unit3_0_scale_shift_tz = { 2, C };
	memory::dims residual_unit3_0_src_tz = { N, C, H, W };

	auto residual_unit3_0_bn_mean_md = memory::desc(residual_unit3_0_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto residual_unit3_0_bn_scale_shift_md = memory::desc(residual_unit3_0_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto residual_unit3_0_src_md = memory::desc(
		residual_unit3_0_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto residual_unit3_0_src_mem = residual_unit1_1_user_conv1_dst_mem; // 
	auto residual_unit3_0_mean_mem = memory(residual_unit3_0_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit3_0_bn_mean.data(), residual_unit3_0_mean_mem);
	auto residual_unit3_0_var_mem = memory(residual_unit3_0_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit3_0_bn_var.data(), residual_unit3_0_var_mem);
	auto residual_unit3_0_scale_shift_mem = memory(residual_unit3_0_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(residual_unit3_0_bn_scale_shift.data(), residual_unit3_0_scale_shift_mem);

	flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto residual_unit3_0_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		residual_unit3_0_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);

	auto residual_unit3_0_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(residual_unit3_0_bnrm_fwd_d, attr1, cpu_engine);
	auto residual_unit3_0_bnrm_fwd = batch_normalization_forward(residual_unit3_0_bnrm_fwd_pd);
	residual_unit3_0_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, residual_unit3_0_src_mem },
			{ MKLDNN_ARG_MEAN, residual_unit3_0_mean_mem },
			{ MKLDNN_ARG_VARIANCE, residual_unit3_0_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, residual_unit3_0_scale_shift_mem },
			{ MKLDNN_ARG_DST, residual_unit1_0_bn_dst_mem }
		}
	);

	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 128;
	IC = C, OC = 128, KH = 3, KW = 3;

	memory::dims  residual_unit3_0_conv_src_tz = { N, C, H, W };
	memory::dims  residual_unit3_0_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims  residual_unit3_0_conv_bias_tz = { OC };
	memory::dims  residual_unit3_0_conv_dst_tz = { N, OC, H, W };


	auto  residual_unit3_0_conv_src_md = memory::desc({ residual_unit3_0_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit3_0_conv_bias_md = memory::desc({ residual_unit3_0_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit3_0_conv_weights_md = memory::desc({ residual_unit3_0_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit3_0_conv_dst_md = memory::desc({ residual_unit3_0_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto residual_unit3_0_user_conv1_weights_md = memory::desc(
		residual_unit3_0_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto residual_unit3_0_user_conv1_bias_md = memory::desc({ residual_unit3_0_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);


	//[Create convolution descriptor]
	auto  residual_unit3_0_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, residual_unit3_0_conv_src_md, residual_unit3_0_conv_weights_md,
		residual_unit3_0_conv_bias_md, residual_unit3_0_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto  residual_unit3_0_conv1_fast_prim_desc = convolution_forward::primitive_desc(residual_unit3_0_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
		reorder(residual_unit1_0_bn_dst_mem, residual_unit1_0_conv1_src_memory)
			.execute(cpu_stream, residual_unit1_0_bn_dst_mem, residual_unit1_0_conv1_src_memory);


	//[Create memory for output]
	// create convolution primitive and add it to net
	auto  residual_unit3_0_fast_conv1 = convolution_forward(residual_unit3_0_conv1_fast_prim_desc);

	residual_unit3_0_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC,  residual_unit1_0_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS,  residual_unit1_0_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, residual_unit1_0_user_conv1_bias_mem },
			{ MKLDNN_ARG_DST,  residual_unit1_0_conv1_dst_memory }
		}
	);

																								   // create reorder between internal and user data if it is needed and
																								   // add it to net after pooling
	if (residual_unit1_0_conv1_dst_memory != residual_unit1_0_user_conv1_dst_mem) {
		reorder(residual_unit1_0_conv1_dst_memory, residual_unit1_0_user_conv1_dst_mem)
			.execute(cpu_stream, residual_unit1_0_conv1_dst_memory, residual_unit1_0_user_conv1_dst_mem);
	}


	/*the 3-1 residual block*/
	/**********************BN***********************************/
	N = 1, H = hei, W = wid, C = 128;
	int residual_unit3_1_bn_mean_size = C;
	int residual_unit3_1_bn_scale_shift_size = 2 * C;

	std::vector<float> residual_unit3_1_bn_mean(residual_unit3_1_bn_mean_size);
	std::vector<float> residual_unit3_1_bn_var(residual_unit3_1_bn_mean_size);
	std::vector<float> residual_unit3_1_bn_scale_shift(residual_unit3_1_bn_scale_shift_size);

	/*residual_block **/
	for (int n = 0; n < residual_unit3_1_bn_mean_size; n++)
	{
		residual_unit3_1_bn_mean[n] = residual_block0->data[1].u[n];
		residual_unit3_1_bn_var[n] = residual_block0->data[1].std[n];
	}
	for (int n = 0; n < residual_unit3_1_bn_scale_shift_size / 2; n++)
	{
		residual_unit3_1_bn_scale_shift[n] = residual_block0->data[1].alpha[n];    //scale
		residual_unit3_1_bn_scale_shift[n + residual_unit3_1_bn_scale_shift_size / 2] = residual_block0->data[1].beta[n];  //shift
	}

	memory::dims residual_unit3_1_mean_tz = { C };
	memory::dims residual_unit3_1_scale_shift_tz = { 2, C };
	memory::dims residual_unit3_1_src_tz = { N, C, H, W };

	auto residual_unit3_1_bn_mean_md = memory::desc(residual_unit3_1_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto residual_unit3_1_bn_scale_shift_md = memory::desc(residual_unit3_1_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto residual_unit3_1_src_md = memory::desc(
		residual_unit3_1_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto residual_unit3_1_src_mem = residual_unit1_0_user_conv1_dst_mem; // 
	auto residual_unit3_1_mean_mem = memory(residual_unit3_1_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit3_1_bn_mean.data(), residual_unit3_1_mean_mem);
	auto residual_unit3_1_var_mem = memory(residual_unit3_1_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit3_1_bn_var.data(), residual_unit3_1_var_mem);
	auto residual_unit3_1_scale_shift_mem = memory(residual_unit3_1_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(residual_unit3_1_bn_scale_shift.data(), residual_unit3_1_scale_shift_mem);

	flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto residual_unit3_1_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		residual_unit3_1_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);

	auto residual_unit3_1_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(residual_unit3_1_bnrm_fwd_d, attr1, cpu_engine);
	auto residual_unit3_1_bnrm_fwd = batch_normalization_forward(residual_unit3_1_bnrm_fwd_pd);
	residual_unit3_1_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, residual_unit3_1_src_mem },
			{ MKLDNN_ARG_MEAN, residual_unit3_1_mean_mem },
			{ MKLDNN_ARG_VARIANCE, residual_unit3_1_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, residual_unit3_1_scale_shift_mem },
			{ MKLDNN_ARG_DST, residual_unit1_1_bn_dst_mem }
		}
	);

	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 128;
	IC = C, OC = 128, KH = 3, KW = 3;


	memory::dims  residual_unit3_1_conv_src_tz = { N, C, H, W };
	memory::dims  residual_unit3_1_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims  residual_unit3_1_conv_bias_tz = { OC };
	memory::dims  residual_unit3_1_conv_dst_tz = { N, OC, H, W };


	auto  residual_unit3_1_conv_src_md = memory::desc({ residual_unit3_1_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit3_1_conv_bias_md = memory::desc({ residual_unit3_1_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit3_1_conv_weights_md = memory::desc({ residual_unit3_1_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit3_1_conv_dst_md = memory::desc({ residual_unit3_1_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto residual_unit3_1_user_conv1_weights_md = memory::desc(
		residual_unit3_1_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto residual_unit3_1_user_conv1_bias_md = memory::desc({ residual_unit3_1_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	//[Create convolution descriptor]
	auto  residual_unit3_1_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, residual_unit3_1_conv_src_md, residual_unit3_1_conv_weights_md,
		residual_unit3_1_conv_bias_md, residual_unit3_1_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto  residual_unit3_1_conv1_fast_prim_desc = convolution_forward::primitive_desc(residual_unit3_1_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
		reorder(residual_unit1_1_bn_dst_mem, residual_unit1_1_conv1_src_memory)
			.execute(cpu_stream, residual_unit1_1_bn_dst_mem, residual_unit1_1_conv1_src_memory);

	//[Create memory for output]
	// create convolution primitive and add it to net
	auto  residual_unit3_1_fast_conv1 = convolution_forward(residual_unit3_1_conv1_fast_prim_desc);

	residual_unit3_1_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC,  residual_unit1_1_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS,  residual_unit1_1_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, residual_unit1_1_user_conv1_bias_mem },
			{ MKLDNN_ARG_DST,  residual_unit1_1_conv1_dst_memory }
		}
	);

	//with sum to the first unit output
	write_to_dnnl_memory(first_unit_conv_output_mem.data(), residual_unit1_1_user_conv1_dst_mem);

	auto residual_unit3_src_reorder_pd = reorder::primitive_desc(residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem, attr2);
	reorder(residual_unit3_src_reorder_pd).execute(cpu_stream, residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem);



	/*the residual_unit4_0 residual block*/
	/**********************BN***********************************/
	N = 1, H = hei, W = wid, C = 128;
	int residual_unit4_0_bn_mean_size = C;
	int residual_unit4_0_bn_scale_shift_size = 2 * C;

	std::vector<float> residual_unit4_0_bn_mean(residual_unit4_0_bn_mean_size);
	std::vector<float> residual_unit4_0_bn_var(residual_unit4_0_bn_mean_size);
	std::vector<float> residual_unit4_0_bn_scale_shift(residual_unit4_0_bn_scale_shift_size);

	/*residual_block **/ residual_block0++;
	for (int n = 0; n < residual_unit4_0_bn_mean_size; n++)
	{
		residual_unit4_0_bn_mean[n] = residual_block0->data[0].u[n];
		residual_unit4_0_bn_var[n] = residual_block0->data[0].std[n];
	}
	for (int n = 0; n < residual_unit4_0_bn_scale_shift_size / 2; n++)
	{
		residual_unit4_0_bn_scale_shift[n] = residual_block0->data[0].alpha[n];    //scale
		residual_unit4_0_bn_scale_shift[n + residual_unit4_0_bn_scale_shift_size / 2] = residual_block0->data[0].beta[n];  //shift
	}

	memory::dims residual_unit4_0_mean_tz = { C };
	memory::dims residual_unit4_0_scale_shift_tz = { 2, C };
	memory::dims residual_unit4_0_src_tz = { N, C, H, W };

	auto residual_unit4_0_bn_mean_md = memory::desc(residual_unit4_0_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto residual_unit4_0_bn_scale_shift_md = memory::desc(residual_unit4_0_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto residual_unit4_0_src_md = memory::desc(
		residual_unit4_0_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto residual_unit4_0_src_mem = residual_unit1_1_user_conv1_dst_mem; // 
	auto residual_unit4_0_mean_mem = memory(residual_unit4_0_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit4_0_bn_mean.data(), residual_unit4_0_mean_mem);
	auto residual_unit4_0_var_mem = memory(residual_unit4_0_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit4_0_bn_var.data(), residual_unit4_0_var_mem);
	auto residual_unit4_0_scale_shift_mem = memory(residual_unit4_0_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(residual_unit4_0_bn_scale_shift.data(), residual_unit4_0_scale_shift_mem);

	flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto residual_unit4_0_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		residual_unit4_0_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);

	auto residual_unit4_0_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(residual_unit4_0_bnrm_fwd_d, attr1, cpu_engine);
	auto residual_unit4_0_bnrm_fwd = batch_normalization_forward(residual_unit4_0_bnrm_fwd_pd);
	residual_unit4_0_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, residual_unit4_0_src_mem },
			{ MKLDNN_ARG_MEAN, residual_unit4_0_mean_mem },
			{ MKLDNN_ARG_VARIANCE, residual_unit4_0_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, residual_unit4_0_scale_shift_mem },
			{ MKLDNN_ARG_DST, residual_unit1_0_bn_dst_mem }
		}
	);

	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 128;
	IC = C, OC = 128, KH = 3, KW = 3;

	memory::dims  residual_unit4_0_conv_src_tz = { N, C, H, W };
	memory::dims  residual_unit4_0_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims  residual_unit4_0_conv_bias_tz = { OC };
	memory::dims  residual_unit4_0_conv_dst_tz = { N, OC, H, W };


	auto  residual_unit4_0_conv_src_md = memory::desc({ residual_unit4_0_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit4_0_conv_bias_md = memory::desc({ residual_unit4_0_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit4_0_conv_weights_md = memory::desc({ residual_unit4_0_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit4_0_conv_dst_md = memory::desc({ residual_unit4_0_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto residual_unit4_0_user_conv1_weights_md = memory::desc(
		residual_unit4_0_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto residual_unit4_0_user_conv1_bias_md = memory::desc({ residual_unit4_0_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	//[Create convolution descriptor]
	auto  residual_unit4_0_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, residual_unit4_0_conv_src_md, residual_unit4_0_conv_weights_md,
		residual_unit4_0_conv_bias_md, residual_unit4_0_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto  residual_unit4_0_conv1_fast_prim_desc = convolution_forward::primitive_desc(residual_unit4_0_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
		reorder(residual_unit1_0_bn_dst_mem, residual_unit1_0_conv1_src_memory)
			.execute(cpu_stream, residual_unit1_0_bn_dst_mem, residual_unit1_0_conv1_src_memory);

	//[Create memory for output]
	// create convolution primitive and add it to net
	auto  residual_unit4_0_fast_conv1 = convolution_forward(residual_unit4_0_conv1_fast_prim_desc);

	residual_unit4_0_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC,  residual_unit1_0_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS,  residual_unit1_0_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, residual_unit1_0_user_conv1_bias_mem },
			{ MKLDNN_ARG_DST,  residual_unit1_0_conv1_dst_memory }
		}
	);


																								   // create reorder between internal and user data if it is needed and
																								   // add it to net after pooling
	if (residual_unit1_0_conv1_dst_memory != residual_unit1_0_user_conv1_dst_mem) {
		reorder(residual_unit1_0_conv1_dst_memory, residual_unit1_0_user_conv1_dst_mem)
			.execute(cpu_stream, residual_unit1_0_conv1_dst_memory, residual_unit1_0_user_conv1_dst_mem);
	}


	/*the residual_unit4_1 residual block*/
	/**********************BN***********************************/
	N = 1, H = hei, W = wid, C = 128;
	int residual_unit4_1_bn_mean_size = C;
	int residual_unit4_1_bn_scale_shift_size = 2 * C;

	std::vector<float> residual_unit4_1_bn_mean(residual_unit4_1_bn_mean_size);
	std::vector<float> residual_unit4_1_bn_var(residual_unit4_1_bn_mean_size);
	std::vector<float> residual_unit4_1_bn_scale_shift(residual_unit4_1_bn_scale_shift_size);

	/*residual_block **/
	for (int n = 0; n < residual_unit4_1_bn_mean_size; n++)
	{
		residual_unit4_1_bn_mean[n] = residual_block0->data[1].u[n];
		residual_unit4_1_bn_var[n] = residual_block0->data[1].std[n];
	}
	for (int n = 0; n < residual_unit4_1_bn_scale_shift_size / 2; n++)
	{
		residual_unit4_1_bn_scale_shift[n] = residual_block0->data[1].alpha[n];    //scale
		residual_unit4_1_bn_scale_shift[n + residual_unit4_1_bn_scale_shift_size / 2] = residual_block0->data[1].beta[n];  //shift
	}

	memory::dims residual_unit4_1_mean_tz = { C };
	memory::dims residual_unit4_1_scale_shift_tz = { 2, C };
	memory::dims residual_unit4_1_src_tz = { N, C, H, W };

	auto residual_unit4_1_bn_mean_md = memory::desc(residual_unit4_1_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto residual_unit4_1_bn_scale_shift_md = memory::desc(residual_unit4_1_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto residual_unit4_1_src_md = memory::desc(
		residual_unit4_1_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto residual_unit4_1_src_mem = residual_unit1_0_user_conv1_dst_mem; // 
	auto residual_unit4_1_mean_mem = memory(residual_unit4_1_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit4_1_bn_mean.data(), residual_unit4_1_mean_mem);
	auto residual_unit4_1_var_mem = memory(residual_unit4_1_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit4_1_bn_var.data(), residual_unit4_1_var_mem);
	auto residual_unit4_1_scale_shift_mem = memory(residual_unit4_1_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(residual_unit4_1_bn_scale_shift.data(), residual_unit4_1_scale_shift_mem);

	flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto residual_unit4_1_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		residual_unit4_1_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);

	auto residual_unit4_1_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(residual_unit4_1_bnrm_fwd_d, attr1, cpu_engine);
	auto residual_unit4_1_bnrm_fwd = batch_normalization_forward(residual_unit4_1_bnrm_fwd_pd);
	residual_unit4_1_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, residual_unit4_1_src_mem },
			{ MKLDNN_ARG_MEAN, residual_unit4_1_mean_mem },
			{ MKLDNN_ARG_VARIANCE, residual_unit4_1_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, residual_unit4_1_scale_shift_mem },
			{ MKLDNN_ARG_DST, residual_unit1_1_bn_dst_mem }
		}
	);

	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 128;
	IC = C, OC = 128, KH = 3, KW = 3;



	memory::dims  residual_unit4_1_conv_src_tz = { N, C, H, W };
	memory::dims  residual_unit4_1_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims  residual_unit4_1_conv_bias_tz = { OC };
	memory::dims  residual_unit4_1_conv_dst_tz = { N, OC, H, W };


	auto  residual_unit4_1_conv_src_md = memory::desc({ residual_unit4_1_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit4_1_conv_bias_md = memory::desc({ residual_unit4_1_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit4_1_conv_weights_md = memory::desc({ residual_unit4_1_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit4_1_conv_dst_md = memory::desc({ residual_unit4_1_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto residual_unit4_1_user_conv1_weights_md = memory::desc(
		residual_unit4_1_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto residual_unit4_1_user_conv1_bias_md = memory::desc({ residual_unit4_1_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	//[Create convolution descriptor]
	auto  residual_unit4_1_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, residual_unit4_1_conv_src_md, residual_unit4_1_conv_weights_md,
		residual_unit4_1_conv_bias_md, residual_unit4_1_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto  residual_unit4_1_conv1_fast_prim_desc = convolution_forward::primitive_desc(residual_unit4_1_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
		reorder(residual_unit1_1_bn_dst_mem, residual_unit1_1_conv1_src_memory)
			.execute(cpu_stream, residual_unit1_1_bn_dst_mem, residual_unit1_1_conv1_src_memory);

	//[Create memory for output]
	// create convolution primitive and add it to net
	auto  residual_unit4_1_fast_conv1 = convolution_forward(residual_unit4_1_conv1_fast_prim_desc);

	residual_unit4_1_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC,  residual_unit1_1_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS,  residual_unit1_1_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, residual_unit1_1_user_conv1_bias_mem },
			{ MKLDNN_ARG_DST,  residual_unit1_1_conv1_dst_memory }
		}
	);

	//with sum to the first unit output
	write_to_dnnl_memory(first_unit_conv_output_mem.data(), residual_unit1_1_user_conv1_dst_mem);

	auto residual_unit4_src_reorder_pd = reorder::primitive_desc(residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem, attr2);
	reorder(residual_unit4_src_reorder_pd).execute(cpu_stream, residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem);

	/*the residual_unit5_0 residual block*/
	/**********************BN***********************************/
	N = 1, H = hei, W = wid, C = 128;
	int residual_unit5_0_bn_mean_size = C;
	int residual_unit5_0_bn_scale_shift_size = 2 * C;

	std::vector<float> residual_unit5_0_bn_mean(residual_unit5_0_bn_mean_size);
	std::vector<float> residual_unit5_0_bn_var(residual_unit5_0_bn_mean_size);
	std::vector<float> residual_unit5_0_bn_scale_shift(residual_unit5_0_bn_scale_shift_size);

	/*residual_block **/ residual_block0++;
	for (int n = 0; n < residual_unit5_0_bn_mean_size; n++)
	{
		residual_unit5_0_bn_mean[n] = residual_block0->data[0].u[n];
		residual_unit5_0_bn_var[n] = residual_block0->data[0].std[n];
	}
	for (int n = 0; n < residual_unit5_0_bn_scale_shift_size / 2; n++)
	{
		residual_unit5_0_bn_scale_shift[n] = residual_block0->data[0].alpha[n];    //scale
		residual_unit5_0_bn_scale_shift[n + residual_unit5_0_bn_scale_shift_size / 2] = residual_block0->data[0].beta[n];  //shift
	}

	memory::dims residual_unit5_0_mean_tz = { C };
	memory::dims residual_unit5_0_scale_shift_tz = { 2, C };
	memory::dims residual_unit5_0_src_tz = { N, C, H, W };

	auto residual_unit5_0_bn_mean_md = memory::desc(residual_unit5_0_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto residual_unit5_0_bn_scale_shift_md = memory::desc(residual_unit5_0_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto residual_unit5_0_src_md = memory::desc(
		residual_unit5_0_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto residual_unit5_0_src_mem = residual_unit1_1_user_conv1_dst_mem; // 
	auto residual_unit5_0_mean_mem = memory(residual_unit5_0_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit5_0_bn_mean.data(), residual_unit5_0_mean_mem);
	auto residual_unit5_0_var_mem = memory(residual_unit5_0_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit5_0_bn_var.data(), residual_unit5_0_var_mem);
	auto residual_unit5_0_scale_shift_mem = memory(residual_unit5_0_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(residual_unit5_0_bn_scale_shift.data(), residual_unit5_0_scale_shift_mem);

	flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto residual_unit5_0_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		residual_unit5_0_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);

	auto residual_unit5_0_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(residual_unit5_0_bnrm_fwd_d, attr1, cpu_engine);
	auto residual_unit5_0_bnrm_fwd = batch_normalization_forward(residual_unit5_0_bnrm_fwd_pd);
	residual_unit5_0_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, residual_unit5_0_src_mem },
			{ MKLDNN_ARG_MEAN, residual_unit5_0_mean_mem },
			{ MKLDNN_ARG_VARIANCE, residual_unit5_0_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, residual_unit5_0_scale_shift_mem },
			{ MKLDNN_ARG_DST, residual_unit1_0_bn_dst_mem }
		}
	);

	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 128;
	IC = C, OC = 128, KH = 3, KW = 3;

	memory::dims  residual_unit5_0_conv_src_tz = { N, C, H, W };
	memory::dims  residual_unit5_0_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims  residual_unit5_0_conv_bias_tz = { OC };
	memory::dims  residual_unit5_0_conv_dst_tz = { N, OC, H, W };


	auto  residual_unit5_0_conv_src_md = memory::desc({ residual_unit5_0_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit5_0_conv_bias_md = memory::desc({ residual_unit5_0_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit5_0_conv_weights_md = memory::desc({ residual_unit5_0_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit5_0_conv_dst_md = memory::desc({ residual_unit5_0_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto residual_unit5_0_user_conv1_weights_md = memory::desc(
		residual_unit5_0_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto residual_unit5_0_user_conv1_bias_md = memory::desc({ residual_unit5_0_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	//[Create convolution descriptor]
	auto  residual_unit5_0_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, residual_unit5_0_conv_src_md, residual_unit5_0_conv_weights_md,
		residual_unit5_0_conv_bias_md, residual_unit5_0_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto  residual_unit5_0_conv1_fast_prim_desc = convolution_forward::primitive_desc(residual_unit5_0_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
		reorder(residual_unit1_0_bn_dst_mem, residual_unit1_0_conv1_src_memory)
			.execute(cpu_stream, residual_unit1_0_bn_dst_mem, residual_unit1_0_conv1_src_memory);

	//[Create memory for output]
	// create convolution primitive and add it to net
	auto  residual_unit5_0_fast_conv1 = convolution_forward(residual_unit5_0_conv1_fast_prim_desc);

	residual_unit5_0_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC,  residual_unit1_0_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS,  residual_unit1_0_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, residual_unit1_0_user_conv1_bias_mem },
			{ MKLDNN_ARG_DST,  residual_unit1_0_conv1_dst_memory }
		}
	);


																								   // create reorder between internal and user data if it is needed and
																								   // add it to net after pooling
	if (residual_unit1_0_conv1_dst_memory != residual_unit1_0_user_conv1_dst_mem) {
		reorder(residual_unit1_0_conv1_dst_memory, residual_unit1_0_user_conv1_dst_mem)
			.execute(cpu_stream, residual_unit1_0_conv1_dst_memory, residual_unit1_0_user_conv1_dst_mem);
	}


	/*the residual_unit5_1 residual block*/
	/**********************BN***********************************/
	N = 1, H = hei, W = wid, C = 128;
	int residual_unit5_1_bn_mean_size = C;
	int residual_unit5_1_bn_scale_shift_size = 2 * C;

	std::vector<float> residual_unit5_1_bn_mean(residual_unit5_1_bn_mean_size);
	std::vector<float> residual_unit5_1_bn_var(residual_unit5_1_bn_mean_size);
	std::vector<float> residual_unit5_1_bn_scale_shift(residual_unit5_1_bn_scale_shift_size);

	/*residual_block **/
	for (int n = 0; n < residual_unit5_1_bn_mean_size; n++)
	{
		residual_unit5_1_bn_mean[n] = residual_block0->data[1].u[n];
		residual_unit5_1_bn_var[n] = residual_block0->data[1].std[n];
	}
	for (int n = 0; n < residual_unit5_1_bn_scale_shift_size / 2; n++)
	{
		residual_unit5_1_bn_scale_shift[n] = residual_block0->data[1].alpha[n];    //scale
		residual_unit5_1_bn_scale_shift[n + residual_unit5_1_bn_scale_shift_size / 2] = residual_block0->data[1].beta[n];  //shift
	}

	memory::dims residual_unit5_1_mean_tz = { C };
	memory::dims residual_unit5_1_scale_shift_tz = { 2, C };
	memory::dims residual_unit5_1_src_tz = { N, C, H, W };

	auto residual_unit5_1_bn_mean_md = memory::desc(residual_unit5_1_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto residual_unit5_1_bn_scale_shift_md = memory::desc(residual_unit5_1_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto residual_unit5_1_src_md = memory::desc(
		residual_unit5_1_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto residual_unit5_1_src_mem = residual_unit1_0_user_conv1_dst_mem; // 
	auto residual_unit5_1_mean_mem = memory(residual_unit5_1_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit5_1_bn_mean.data(), residual_unit5_1_mean_mem);
	auto residual_unit5_1_var_mem = memory(residual_unit5_1_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit5_1_bn_var.data(), residual_unit5_1_var_mem);
	auto residual_unit5_1_scale_shift_mem = memory(residual_unit5_1_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(residual_unit5_1_bn_scale_shift.data(), residual_unit5_1_scale_shift_mem);

	flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto residual_unit5_1_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		residual_unit5_1_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);

	auto residual_unit5_1_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(residual_unit5_1_bnrm_fwd_d, attr1, cpu_engine);
	auto residual_unit5_1_bnrm_fwd = batch_normalization_forward(residual_unit5_1_bnrm_fwd_pd);
	residual_unit5_1_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, residual_unit5_1_src_mem },
			{ MKLDNN_ARG_MEAN, residual_unit5_1_mean_mem },
			{ MKLDNN_ARG_VARIANCE, residual_unit5_1_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, residual_unit5_1_scale_shift_mem },
			{ MKLDNN_ARG_DST, residual_unit1_1_bn_dst_mem }
		}
	);

	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 128;
	IC = C, OC = 128, KH = 3, KW = 3;


	memory::dims  residual_unit5_1_conv_src_tz = { N, C, H, W };
	memory::dims  residual_unit5_1_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims  residual_unit5_1_conv_bias_tz = { OC };
	memory::dims  residual_unit5_1_conv_dst_tz = { N, OC, H, W };


	auto  residual_unit5_1_conv_src_md = memory::desc({ residual_unit5_1_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit5_1_conv_bias_md = memory::desc({ residual_unit5_1_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit5_1_conv_weights_md = memory::desc({ residual_unit5_1_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit5_1_conv_dst_md = memory::desc({ residual_unit5_1_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto residual_unit5_1_user_conv1_weights_md = memory::desc(
		residual_unit5_1_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto residual_unit5_1_user_conv1_bias_md = memory::desc({ residual_unit5_1_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	//[Create convolution descriptor]
	auto  residual_unit5_1_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, residual_unit5_1_conv_src_md, residual_unit5_1_conv_weights_md,
		residual_unit5_1_conv_bias_md, residual_unit5_1_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto  residual_unit5_1_conv1_fast_prim_desc = convolution_forward::primitive_desc(residual_unit5_1_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
		reorder(residual_unit1_1_bn_dst_mem, residual_unit1_1_conv1_src_memory)
			.execute(cpu_stream, residual_unit1_1_bn_dst_mem, residual_unit1_1_conv1_src_memory);

	//[Create memory for output]
	// create convolution primitive and add it to net
	auto  residual_unit5_1_fast_conv1 = convolution_forward(residual_unit5_1_conv1_fast_prim_desc);

	residual_unit5_1_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC,  residual_unit1_1_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS,  residual_unit1_1_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, residual_unit1_1_user_conv1_bias_mem },
			{ MKLDNN_ARG_DST,  residual_unit1_1_conv1_dst_memory }
		}
	);

	//with sum to the first unit output
	write_to_dnnl_memory(first_unit_conv_output_mem.data(), residual_unit1_1_user_conv1_dst_mem);

	auto residual_unit5_src_reorder_pd = reorder::primitive_desc(residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem, attr2);
	reorder(residual_unit5_src_reorder_pd).execute(cpu_stream, residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem);

	/*the residual_unit6_0 residual block*/
	/**********************BN***********************************/
	N = 1, H = hei, W = wid, C = 128;
	int residual_unit6_0_bn_mean_size = C;
	int residual_unit6_0_bn_scale_shift_size = 2 * C;

	std::vector<float> residual_unit6_0_bn_mean(residual_unit6_0_bn_mean_size);
	std::vector<float> residual_unit6_0_bn_var(residual_unit6_0_bn_mean_size);
	std::vector<float> residual_unit6_0_bn_scale_shift(residual_unit6_0_bn_scale_shift_size);

	/*residual_block **/ residual_block0++;
	for (int n = 0; n < residual_unit6_0_bn_mean_size; n++)
	{
		residual_unit6_0_bn_mean[n] = residual_block0->data[0].u[n];
		residual_unit6_0_bn_var[n] = residual_block0->data[0].std[n];
	}
	for (int n = 0; n < residual_unit6_0_bn_scale_shift_size / 2; n++)
	{
		residual_unit6_0_bn_scale_shift[n] = residual_block0->data[0].alpha[n];    //scale
		residual_unit6_0_bn_scale_shift[n + residual_unit6_0_bn_scale_shift_size / 2] = residual_block0->data[0].beta[n];  //shift
	}

	memory::dims residual_unit6_0_mean_tz = { C };
	memory::dims residual_unit6_0_scale_shift_tz = { 2, C };
	memory::dims residual_unit6_0_src_tz = { N, C, H, W };

	auto residual_unit6_0_bn_mean_md = memory::desc(residual_unit6_0_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto residual_unit6_0_bn_scale_shift_md = memory::desc(residual_unit6_0_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto residual_unit6_0_src_md = memory::desc(
		residual_unit6_0_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto residual_unit6_0_src_mem = residual_unit1_1_user_conv1_dst_mem; // 
	auto residual_unit6_0_mean_mem = memory(residual_unit6_0_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit6_0_bn_mean.data(), residual_unit6_0_mean_mem);
	auto residual_unit6_0_var_mem = memory(residual_unit6_0_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit6_0_bn_var.data(), residual_unit6_0_var_mem);
	auto residual_unit6_0_scale_shift_mem = memory(residual_unit6_0_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(residual_unit6_0_bn_scale_shift.data(), residual_unit6_0_scale_shift_mem);

	flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto residual_unit6_0_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		residual_unit6_0_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);

	auto residual_unit6_0_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(residual_unit6_0_bnrm_fwd_d, attr1, cpu_engine);
	auto residual_unit6_0_bnrm_fwd = batch_normalization_forward(residual_unit6_0_bnrm_fwd_pd);
	residual_unit6_0_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, residual_unit6_0_src_mem },
			{ MKLDNN_ARG_MEAN, residual_unit6_0_mean_mem },
			{ MKLDNN_ARG_VARIANCE, residual_unit6_0_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, residual_unit6_0_scale_shift_mem },
			{ MKLDNN_ARG_DST, residual_unit1_0_bn_dst_mem }
		}
	);

	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 128;
	IC = C, OC = 128, KH = 3, KW = 3;

	memory::dims  residual_unit6_0_conv_src_tz = { N, C, H, W };
	memory::dims  residual_unit6_0_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims  residual_unit6_0_conv_bias_tz = { OC };
	memory::dims  residual_unit6_0_conv_dst_tz = { N, OC, H, W };


	auto  residual_unit6_0_conv_src_md = memory::desc({ residual_unit6_0_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit6_0_conv_bias_md = memory::desc({ residual_unit6_0_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit6_0_conv_weights_md = memory::desc({ residual_unit6_0_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit6_0_conv_dst_md = memory::desc({ residual_unit6_0_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto residual_unit6_0_user_conv1_weights_md = memory::desc(
		residual_unit6_0_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto residual_unit6_0_user_conv1_bias_md = memory::desc({ residual_unit6_0_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	//[Create convolution descriptor]
	auto  residual_unit6_0_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, residual_unit6_0_conv_src_md, residual_unit6_0_conv_weights_md,
		residual_unit6_0_conv_bias_md, residual_unit6_0_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto  residual_unit6_0_conv1_fast_prim_desc = convolution_forward::primitive_desc(residual_unit6_0_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
		reorder(residual_unit1_0_bn_dst_mem, residual_unit1_0_conv1_src_memory)
			.execute(cpu_stream, residual_unit1_0_bn_dst_mem, residual_unit1_0_conv1_src_memory);

	//[Create memory for output]
	// create convolution primitive and add it to net
	auto  residual_unit6_0_fast_conv1 = convolution_forward(residual_unit6_0_conv1_fast_prim_desc);

	residual_unit6_0_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC,  residual_unit1_0_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS,  residual_unit1_0_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, residual_unit1_0_user_conv1_bias_mem },
			{ MKLDNN_ARG_DST,  residual_unit1_0_conv1_dst_memory }
		}
	);


																								   // create reorder between internal and user data if it is needed and
																								   // add it to net after pooling
	if (residual_unit1_0_conv1_dst_memory != residual_unit1_0_user_conv1_dst_mem) {
		reorder(residual_unit1_0_conv1_dst_memory, residual_unit1_0_user_conv1_dst_mem)
			.execute(cpu_stream, residual_unit1_0_conv1_dst_memory, residual_unit1_0_user_conv1_dst_mem);
	}


	/*the residual_unit6_1 residual block*/
	/**********************BN***********************************/
	N = 1, H = hei, W = wid, C = 128;
	int residual_unit6_1_bn_mean_size = C;
	int residual_unit6_1_bn_scale_shift_size = 2 * C;

	std::vector<float> residual_unit6_1_bn_mean(residual_unit6_1_bn_mean_size);
	std::vector<float> residual_unit6_1_bn_var(residual_unit6_1_bn_mean_size);
	std::vector<float> residual_unit6_1_bn_scale_shift(residual_unit6_1_bn_scale_shift_size);

	/*residual_block **/
	for (int n = 0; n < residual_unit6_1_bn_mean_size; n++)
	{
		residual_unit6_1_bn_mean[n] = residual_block0->data[1].u[n];
		residual_unit6_1_bn_var[n] = residual_block0->data[1].std[n];
	}
	for (int n = 0; n < residual_unit6_1_bn_scale_shift_size / 2; n++)
	{
		residual_unit6_1_bn_scale_shift[n] = residual_block0->data[1].alpha[n];    //scale
		residual_unit6_1_bn_scale_shift[n + residual_unit6_1_bn_scale_shift_size / 2] = residual_block0->data[1].beta[n];  //shift
	}

	memory::dims residual_unit6_1_mean_tz = { C };
	memory::dims residual_unit6_1_scale_shift_tz = { 2, C };
	memory::dims residual_unit6_1_src_tz = { N, C, H, W };

	auto residual_unit6_1_bn_mean_md = memory::desc(residual_unit6_1_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto residual_unit6_1_bn_scale_shift_md = memory::desc(residual_unit6_1_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto residual_unit6_1_src_md = memory::desc(
		residual_unit6_1_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto residual_unit6_1_src_mem = residual_unit1_0_user_conv1_dst_mem; // 
	auto residual_unit6_1_mean_mem = memory(residual_unit6_1_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit6_1_bn_mean.data(), residual_unit6_1_mean_mem);
	auto residual_unit6_1_var_mem = memory(residual_unit6_1_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit6_1_bn_var.data(), residual_unit6_1_var_mem);
	auto residual_unit6_1_scale_shift_mem = memory(residual_unit6_1_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(residual_unit6_1_bn_scale_shift.data(), residual_unit6_1_scale_shift_mem);

	flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto residual_unit6_1_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		residual_unit6_1_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);

	auto residual_unit6_1_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(residual_unit6_1_bnrm_fwd_d, attr1, cpu_engine);
	auto residual_unit6_1_bnrm_fwd = batch_normalization_forward(residual_unit6_1_bnrm_fwd_pd);
	residual_unit6_1_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, residual_unit6_1_src_mem },
			{ MKLDNN_ARG_MEAN, residual_unit6_1_mean_mem },
			{ MKLDNN_ARG_VARIANCE, residual_unit6_1_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, residual_unit6_1_scale_shift_mem },
			{ MKLDNN_ARG_DST, residual_unit1_1_bn_dst_mem }
		}
	);

	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 128;
	IC = C, OC = 128, KH = 3, KW = 3;

	memory::dims  residual_unit6_1_conv_src_tz = { N, C, H, W };
	memory::dims  residual_unit6_1_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims  residual_unit6_1_conv_bias_tz = { OC };
	memory::dims  residual_unit6_1_conv_dst_tz = { N, OC, H, W };


	auto  residual_unit6_1_conv_src_md = memory::desc({ residual_unit6_1_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit6_1_conv_bias_md = memory::desc({ residual_unit6_1_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit6_1_conv_weights_md = memory::desc({ residual_unit6_1_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit6_1_conv_dst_md = memory::desc({ residual_unit6_1_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto residual_unit6_1_user_conv1_weights_md = memory::desc(
		residual_unit6_1_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto residual_unit6_1_user_conv1_bias_md = memory::desc({ residual_unit6_1_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	//[Create convolution descriptor]
	auto  residual_unit6_1_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, residual_unit6_1_conv_src_md, residual_unit6_1_conv_weights_md,
		residual_unit6_1_conv_bias_md, residual_unit6_1_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto  residual_unit6_1_conv1_fast_prim_desc = convolution_forward::primitive_desc(residual_unit6_1_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
		reorder(residual_unit1_1_bn_dst_mem, residual_unit1_1_conv1_src_memory)
			.execute(cpu_stream, residual_unit1_1_bn_dst_mem, residual_unit1_1_conv1_src_memory);

	//[Create memory for output]
	// create convolution primitive and add it to net
	auto  residual_unit6_1_fast_conv1 = convolution_forward(residual_unit6_1_conv1_fast_prim_desc);

	residual_unit6_1_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC,  residual_unit1_1_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS,  residual_unit1_1_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, residual_unit1_1_user_conv1_bias_mem },
			{ MKLDNN_ARG_DST,  residual_unit1_1_conv1_dst_memory }
		}
	);

	//with sum to the first unit output
	write_to_dnnl_memory(first_unit_conv_output_mem.data(), residual_unit1_1_user_conv1_dst_mem);

	auto residual_unit6_src_reorder_pd = reorder::primitive_desc(residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem, attr2);
	reorder(residual_unit6_src_reorder_pd).execute(cpu_stream, residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem);

	/*the residual_unit7_0 residual block*/
	/**********************BN***********************************/
	N = 1, H = hei, W = wid, C = 128;
	int residual_unit7_0_bn_mean_size = C;
	int residual_unit7_0_bn_scale_shift_size = 2 * C;

	std::vector<float> residual_unit7_0_bn_mean(residual_unit7_0_bn_mean_size);
	std::vector<float> residual_unit7_0_bn_var(residual_unit7_0_bn_mean_size);
	std::vector<float> residual_unit7_0_bn_scale_shift(residual_unit7_0_bn_scale_shift_size);

	/*residual_block **/ residual_block0++;
	for (int n = 0; n < residual_unit7_0_bn_mean_size; n++)
	{
		residual_unit7_0_bn_mean[n] = residual_block0->data[0].u[n];
		residual_unit7_0_bn_var[n] = residual_block0->data[0].std[n];
	}
	for (int n = 0; n < residual_unit7_0_bn_scale_shift_size / 2; n++)
	{
		residual_unit7_0_bn_scale_shift[n] = residual_block0->data[0].alpha[n];    //scale
		residual_unit7_0_bn_scale_shift[n + residual_unit7_0_bn_scale_shift_size / 2] = residual_block0->data[0].beta[n];  //shift
	}

	memory::dims residual_unit7_0_mean_tz = { C };
	memory::dims residual_unit7_0_scale_shift_tz = { 2, C };
	memory::dims residual_unit7_0_src_tz = { N, C, H, W };

	auto residual_unit7_0_bn_mean_md = memory::desc(residual_unit7_0_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto residual_unit7_0_bn_scale_shift_md = memory::desc(residual_unit7_0_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto residual_unit7_0_src_md = memory::desc(
		residual_unit7_0_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto residual_unit7_0_src_mem = residual_unit1_1_user_conv1_dst_mem; // 
	auto residual_unit7_0_mean_mem = memory(residual_unit7_0_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit7_0_bn_mean.data(), residual_unit7_0_mean_mem);
	auto residual_unit7_0_var_mem = memory(residual_unit7_0_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit7_0_bn_var.data(), residual_unit7_0_var_mem);
	auto residual_unit7_0_scale_shift_mem = memory(residual_unit7_0_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(residual_unit7_0_bn_scale_shift.data(), residual_unit7_0_scale_shift_mem);

	flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto residual_unit7_0_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		residual_unit7_0_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);

	auto residual_unit7_0_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(residual_unit7_0_bnrm_fwd_d, attr1, cpu_engine);
	auto residual_unit7_0_bnrm_fwd = batch_normalization_forward(residual_unit7_0_bnrm_fwd_pd);
	residual_unit7_0_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, residual_unit7_0_src_mem },
			{ MKLDNN_ARG_MEAN, residual_unit7_0_mean_mem },
			{ MKLDNN_ARG_VARIANCE, residual_unit7_0_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, residual_unit7_0_scale_shift_mem },
			{ MKLDNN_ARG_DST, residual_unit1_0_bn_dst_mem }
		}
	);

	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 128;
	IC = C, OC = 128, KH = 3, KW = 3;


	memory::dims  residual_unit7_0_conv_src_tz = { N, C, H, W };
	memory::dims  residual_unit7_0_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims  residual_unit7_0_conv_bias_tz = { OC };
	memory::dims  residual_unit7_0_conv_dst_tz = { N, OC, H, W };


	auto  residual_unit7_0_conv_src_md = memory::desc({ residual_unit7_0_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit7_0_conv_bias_md = memory::desc({ residual_unit7_0_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit7_0_conv_weights_md = memory::desc({ residual_unit7_0_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit7_0_conv_dst_md = memory::desc({ residual_unit7_0_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto residual_unit7_0_user_conv1_weights_md = memory::desc(
		residual_unit7_0_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto residual_unit7_0_user_conv1_bias_md = memory::desc({ residual_unit7_0_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	//[Create convolution descriptor]
	auto  residual_unit7_0_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, residual_unit7_0_conv_src_md, residual_unit7_0_conv_weights_md,
		residual_unit7_0_conv_bias_md, residual_unit7_0_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto  residual_unit7_0_conv1_fast_prim_desc = convolution_forward::primitive_desc(residual_unit7_0_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
		reorder(residual_unit1_0_bn_dst_mem, residual_unit1_0_conv1_src_memory)
			.execute(cpu_stream, residual_unit1_0_bn_dst_mem, residual_unit1_0_conv1_src_memory);

	//[Create memory for output]
	// create convolution primitive and add it to net
	auto  residual_unit7_0_fast_conv1 = convolution_forward(residual_unit7_0_conv1_fast_prim_desc);

	residual_unit7_0_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC,  residual_unit1_0_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS,  residual_unit1_0_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, residual_unit1_0_user_conv1_bias_mem },
			{ MKLDNN_ARG_DST,  residual_unit1_0_conv1_dst_memory }
		}
	);


																								   // create reorder between internal and user data if it is needed and
																								   // add it to net after pooling
	if (residual_unit1_0_conv1_dst_memory != residual_unit1_0_user_conv1_dst_mem) {
		reorder(residual_unit1_0_conv1_dst_memory, residual_unit1_0_user_conv1_dst_mem)
			.execute(cpu_stream, residual_unit1_0_conv1_dst_memory, residual_unit1_0_user_conv1_dst_mem);
	}


	/*the residual_unit7_1 residual block*/
	/**********************BN***********************************/
	N = 1, H = hei, W = wid, C = 128;
	int residual_unit7_1_bn_mean_size = C;
	int residual_unit7_1_bn_scale_shift_size = 2 * C;

	std::vector<float> residual_unit7_1_bn_mean(residual_unit7_1_bn_mean_size);
	std::vector<float> residual_unit7_1_bn_var(residual_unit7_1_bn_mean_size);
	std::vector<float> residual_unit7_1_bn_scale_shift(residual_unit7_1_bn_scale_shift_size);

	/*residual_block **/
	for (int n = 0; n < residual_unit7_1_bn_mean_size; n++)
	{
		residual_unit7_1_bn_mean[n] = residual_block0->data[1].u[n];
		residual_unit7_1_bn_var[n] = residual_block0->data[1].std[n];
	}
	for (int n = 0; n < residual_unit7_1_bn_scale_shift_size / 2; n++)
	{
		residual_unit7_1_bn_scale_shift[n] = residual_block0->data[1].alpha[n];    //scale
		residual_unit7_1_bn_scale_shift[n + residual_unit7_1_bn_scale_shift_size / 2] = residual_block0->data[1].beta[n];  //shift
	}

	memory::dims residual_unit7_1_mean_tz = { C };
	memory::dims residual_unit7_1_scale_shift_tz = { 2, C };
	memory::dims residual_unit7_1_src_tz = { N, C, H, W };

	auto residual_unit7_1_bn_mean_md = memory::desc(residual_unit7_1_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto residual_unit7_1_bn_scale_shift_md = memory::desc(residual_unit7_1_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto residual_unit7_1_src_md = memory::desc(
		residual_unit7_1_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto residual_unit7_1_src_mem = residual_unit1_0_user_conv1_dst_mem; // 
	auto residual_unit7_1_mean_mem = memory(residual_unit7_1_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit7_1_bn_mean.data(), residual_unit7_1_mean_mem);
	auto residual_unit7_1_var_mem = memory(residual_unit7_1_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit7_1_bn_var.data(), residual_unit7_1_var_mem);
	auto residual_unit7_1_scale_shift_mem = memory(residual_unit7_1_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(residual_unit7_1_bn_scale_shift.data(), residual_unit7_1_scale_shift_mem);

	flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto residual_unit7_1_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		residual_unit7_1_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);

	auto residual_unit7_1_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(residual_unit7_1_bnrm_fwd_d, attr1, cpu_engine);
	auto residual_unit7_1_bnrm_fwd = batch_normalization_forward(residual_unit7_1_bnrm_fwd_pd);
	residual_unit7_1_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, residual_unit7_1_src_mem },
			{ MKLDNN_ARG_MEAN, residual_unit7_1_mean_mem },
			{ MKLDNN_ARG_VARIANCE, residual_unit7_1_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, residual_unit7_1_scale_shift_mem },
			{ MKLDNN_ARG_DST, residual_unit1_1_bn_dst_mem }
		}
	);

	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 128;
	IC = C, OC = 128, KH = 3, KW = 3;


	memory::dims  residual_unit7_1_conv_src_tz = { N, C, H, W };
	memory::dims  residual_unit7_1_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims  residual_unit7_1_conv_bias_tz = { OC };
	memory::dims  residual_unit7_1_conv_dst_tz = { N, OC, H, W };


	auto  residual_unit7_1_conv_src_md = memory::desc({ residual_unit7_1_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit7_1_conv_bias_md = memory::desc({ residual_unit7_1_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit7_1_conv_weights_md = memory::desc({ residual_unit7_1_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit7_1_conv_dst_md = memory::desc({ residual_unit7_1_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto residual_unit7_1_user_conv1_weights_md = memory::desc(
		residual_unit7_1_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto residual_unit7_1_user_conv1_bias_md = memory::desc({ residual_unit7_1_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	//[Create convolution descriptor]
	auto  residual_unit7_1_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, residual_unit7_1_conv_src_md, residual_unit7_1_conv_weights_md,
		residual_unit7_1_conv_bias_md, residual_unit7_1_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto  residual_unit7_1_conv1_fast_prim_desc = convolution_forward::primitive_desc(residual_unit7_1_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
		reorder(residual_unit1_1_bn_dst_mem, residual_unit1_1_conv1_src_memory)
			.execute(cpu_stream, residual_unit1_1_bn_dst_mem, residual_unit1_1_conv1_src_memory);

	//[Create memory for output]
	// create convolution primitive and add it to net
	auto  residual_unit7_1_fast_conv1 = convolution_forward(residual_unit7_1_conv1_fast_prim_desc);

	residual_unit7_1_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC,  residual_unit1_1_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS,  residual_unit1_1_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, residual_unit1_1_user_conv1_bias_mem },
			{ MKLDNN_ARG_DST,  residual_unit1_1_conv1_dst_memory }
		}
	);

	//with sum to the first unit output
	write_to_dnnl_memory(first_unit_conv_output_mem.data(), residual_unit1_1_user_conv1_dst_mem);

	auto residual_unit7_src_reorder_pd = reorder::primitive_desc(residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem, attr2);
	reorder(residual_unit7_src_reorder_pd).execute(cpu_stream, residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem);

	/*the residual_unit8_0 residual block*/
	/**********************BN***********************************/
	N = 1, H = hei, W = wid, C = 128;
	int residual_unit8_0_bn_mean_size = C;
	int residual_unit8_0_bn_scale_shift_size = 2 * C;

	std::vector<float> residual_unit8_0_bn_mean(residual_unit8_0_bn_mean_size);
	std::vector<float> residual_unit8_0_bn_var(residual_unit8_0_bn_mean_size);
	std::vector<float> residual_unit8_0_bn_scale_shift(residual_unit8_0_bn_scale_shift_size);

	/*residual_block **/ residual_block0++;
	for (int n = 0; n < residual_unit8_0_bn_mean_size; n++)
	{
		residual_unit8_0_bn_mean[n] = residual_block0->data[0].u[n];
		residual_unit8_0_bn_var[n] = residual_block0->data[0].std[n];
	}
	for (int n = 0; n < residual_unit8_0_bn_scale_shift_size / 2; n++)
	{
		residual_unit8_0_bn_scale_shift[n] = residual_block0->data[0].alpha[n];    //scale
		residual_unit8_0_bn_scale_shift[n + residual_unit8_0_bn_scale_shift_size / 2] = residual_block0->data[0].beta[n];  //shift
	}

	memory::dims residual_unit8_0_mean_tz = { C };
	memory::dims residual_unit8_0_scale_shift_tz = { 2, C };
	memory::dims residual_unit8_0_src_tz = { N, C, H, W };

	auto residual_unit8_0_bn_mean_md = memory::desc(residual_unit8_0_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto residual_unit8_0_bn_scale_shift_md = memory::desc(residual_unit8_0_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto residual_unit8_0_src_md = memory::desc(
		residual_unit8_0_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto residual_unit8_0_src_mem = residual_unit1_1_user_conv1_dst_mem; // 
	auto residual_unit8_0_mean_mem = memory(residual_unit8_0_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit8_0_bn_mean.data(), residual_unit8_0_mean_mem);
	auto residual_unit8_0_var_mem = memory(residual_unit8_0_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit8_0_bn_var.data(), residual_unit8_0_var_mem);
	auto residual_unit8_0_scale_shift_mem = memory(residual_unit8_0_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(residual_unit8_0_bn_scale_shift.data(), residual_unit8_0_scale_shift_mem);

	flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto residual_unit8_0_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		residual_unit8_0_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);

	auto residual_unit8_0_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(residual_unit8_0_bnrm_fwd_d, attr1, cpu_engine);
	auto residual_unit8_0_bnrm_fwd = batch_normalization_forward(residual_unit8_0_bnrm_fwd_pd);
	residual_unit8_0_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, residual_unit8_0_src_mem },
			{ MKLDNN_ARG_MEAN, residual_unit8_0_mean_mem },
			{ MKLDNN_ARG_VARIANCE, residual_unit8_0_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, residual_unit8_0_scale_shift_mem },
			{ MKLDNN_ARG_DST, residual_unit1_0_bn_dst_mem }
		}
	);

	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 128;
	IC = C, OC = 128, KH = 3, KW = 3;


	memory::dims  residual_unit8_0_conv_src_tz = { N, C, H, W };
	memory::dims  residual_unit8_0_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims  residual_unit8_0_conv_bias_tz = { OC };
	memory::dims  residual_unit8_0_conv_dst_tz = { N, OC, H, W };


	auto  residual_unit8_0_conv_src_md = memory::desc({ residual_unit8_0_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit8_0_conv_bias_md = memory::desc({ residual_unit8_0_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit8_0_conv_weights_md = memory::desc({ residual_unit8_0_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit8_0_conv_dst_md = memory::desc({ residual_unit8_0_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto residual_unit8_0_user_conv1_weights_md = memory::desc(
		residual_unit8_0_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto residual_unit8_0_user_conv1_bias_md = memory::desc({ residual_unit8_0_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	//[Create convolution descriptor]
	auto  residual_unit8_0_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, residual_unit8_0_conv_src_md, residual_unit8_0_conv_weights_md,
		residual_unit8_0_conv_bias_md, residual_unit8_0_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto  residual_unit8_0_conv1_fast_prim_desc = convolution_forward::primitive_desc(residual_unit8_0_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
		reorder(residual_unit1_0_bn_dst_mem, residual_unit1_0_conv1_src_memory)
			.execute(cpu_stream, residual_unit1_0_bn_dst_mem, residual_unit1_0_conv1_src_memory);

	//[Create memory for output]
	// create convolution primitive and add it to net
	auto  residual_unit8_0_fast_conv1 = convolution_forward(residual_unit8_0_conv1_fast_prim_desc);

	residual_unit8_0_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC,  residual_unit1_0_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS,  residual_unit1_0_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, residual_unit1_0_user_conv1_bias_mem },
			{ MKLDNN_ARG_DST,  residual_unit1_0_conv1_dst_memory }
		}
	);


																								   // create reorder between internal and user data if it is needed and
																								   // add it to net after pooling
	if (residual_unit1_0_conv1_dst_memory != residual_unit1_0_user_conv1_dst_mem) {
		reorder(residual_unit1_0_conv1_dst_memory, residual_unit1_0_user_conv1_dst_mem)
			.execute(cpu_stream, residual_unit1_0_conv1_dst_memory, residual_unit1_0_user_conv1_dst_mem);
	}


	/*the residual_unit8_1 residual block*/
	/**********************BN***********************************/
	N = 1, H = hei, W = wid, C = 128;
	int residual_unit8_1_bn_mean_size = C;
	int residual_unit8_1_bn_scale_shift_size = 2 * C;

	std::vector<float> residual_unit8_1_bn_mean(residual_unit8_1_bn_mean_size);
	std::vector<float> residual_unit8_1_bn_var(residual_unit8_1_bn_mean_size);
	std::vector<float> residual_unit8_1_bn_scale_shift(residual_unit8_1_bn_scale_shift_size);

	/*residual_block **/
	for (int n = 0; n < residual_unit8_1_bn_mean_size; n++)
	{
		residual_unit8_1_bn_mean[n] = residual_block0->data[1].u[n];
		residual_unit8_1_bn_var[n] = residual_block0->data[1].std[n];
	}
	for (int n = 0; n < residual_unit8_1_bn_scale_shift_size / 2; n++)
	{
		residual_unit8_1_bn_scale_shift[n] = residual_block0->data[1].alpha[n];    //scale
		residual_unit8_1_bn_scale_shift[n + residual_unit8_1_bn_scale_shift_size / 2] = residual_block0->data[1].beta[n];  //shift
	}

	memory::dims residual_unit8_1_mean_tz = { C };
	memory::dims residual_unit8_1_scale_shift_tz = { 2, C };
	memory::dims residual_unit8_1_src_tz = { N, C, H, W };

	auto residual_unit8_1_bn_mean_md = memory::desc(residual_unit8_1_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto residual_unit8_1_bn_scale_shift_md = memory::desc(residual_unit8_1_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto residual_unit8_1_src_md = memory::desc(
		residual_unit8_1_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto residual_unit8_1_src_mem = residual_unit1_0_user_conv1_dst_mem; // 
	auto residual_unit8_1_mean_mem = memory(residual_unit8_1_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit8_1_bn_mean.data(), residual_unit8_1_mean_mem);
	auto residual_unit8_1_var_mem = memory(residual_unit8_1_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit8_1_bn_var.data(), residual_unit8_1_var_mem);
	auto residual_unit8_1_scale_shift_mem = memory(residual_unit8_1_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(residual_unit8_1_bn_scale_shift.data(), residual_unit8_1_scale_shift_mem);

	flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto residual_unit8_1_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		residual_unit8_1_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);

	auto residual_unit8_1_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(residual_unit8_1_bnrm_fwd_d, attr1, cpu_engine);
	auto residual_unit8_1_bnrm_fwd = batch_normalization_forward(residual_unit8_1_bnrm_fwd_pd);
	residual_unit8_1_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, residual_unit8_1_src_mem },
			{ MKLDNN_ARG_MEAN, residual_unit8_1_mean_mem },
			{ MKLDNN_ARG_VARIANCE, residual_unit8_1_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, residual_unit8_1_scale_shift_mem },
			{ MKLDNN_ARG_DST, residual_unit1_1_bn_dst_mem }
		}
	);

	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 128;
	IC = C, OC = 128, KH = 3, KW = 3;


	memory::dims  residual_unit8_1_conv_src_tz = { N, C, H, W };
	memory::dims  residual_unit8_1_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims  residual_unit8_1_conv_bias_tz = { OC };
	memory::dims  residual_unit8_1_conv_dst_tz = { N, OC, H, W };


	auto  residual_unit8_1_conv_src_md = memory::desc({ residual_unit8_1_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit8_1_conv_bias_md = memory::desc({ residual_unit8_1_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit8_1_conv_weights_md = memory::desc({ residual_unit8_1_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit8_1_conv_dst_md = memory::desc({ residual_unit8_1_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto residual_unit8_1_user_conv1_weights_md = memory::desc(
		residual_unit8_1_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto residual_unit8_1_user_conv1_bias_md = memory::desc({ residual_unit8_1_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	//[Create convolution descriptor]
	auto  residual_unit8_1_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, residual_unit8_1_conv_src_md, residual_unit8_1_conv_weights_md,
		residual_unit8_1_conv_bias_md, residual_unit8_1_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto  residual_unit8_1_conv1_fast_prim_desc = convolution_forward::primitive_desc(residual_unit8_1_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
		reorder(residual_unit1_1_bn_dst_mem, residual_unit1_1_conv1_src_memory)
			.execute(cpu_stream, residual_unit1_1_bn_dst_mem, residual_unit1_1_conv1_src_memory);

	//[Create memory for output]
	// create convolution primitive and add it to net
	auto  residual_unit8_1_fast_conv1 = convolution_forward(residual_unit8_1_conv1_fast_prim_desc);

	residual_unit8_1_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC,  residual_unit1_1_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS,  residual_unit1_1_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, residual_unit1_1_user_conv1_bias_mem },
			{ MKLDNN_ARG_DST,  residual_unit1_1_conv1_dst_memory }
		}
	);

	//with sum to the first unit output
	write_to_dnnl_memory(first_unit_conv_output_mem.data(), residual_unit1_1_user_conv1_dst_mem);

	auto residual_unit8_src_reorder_pd = reorder::primitive_desc(residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem, attr2);
	reorder(residual_unit8_src_reorder_pd).execute(cpu_stream, residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem);

	/*the residual_unit9_0 residual block*/
	/**********************BN***********************************/
	N = 1, H = hei, W = wid, C = 128;
	int residual_unit9_0_bn_mean_size = C;
	int residual_unit9_0_bn_scale_shift_size = 2 * C;

	std::vector<float> residual_unit9_0_bn_mean(residual_unit9_0_bn_mean_size);
	std::vector<float> residual_unit9_0_bn_var(residual_unit9_0_bn_mean_size);
	std::vector<float> residual_unit9_0_bn_scale_shift(residual_unit9_0_bn_scale_shift_size);

	/*residual_block **/ residual_block0++;
	for (int n = 0; n < residual_unit9_0_bn_mean_size; n++)
	{
		residual_unit9_0_bn_mean[n] = residual_block0->data[0].u[n];
		residual_unit9_0_bn_var[n] = residual_block0->data[0].std[n];
	}
	for (int n = 0; n < residual_unit9_0_bn_scale_shift_size / 2; n++)
	{
		residual_unit9_0_bn_scale_shift[n] = residual_block0->data[0].alpha[n];    //scale
		residual_unit9_0_bn_scale_shift[n + residual_unit9_0_bn_scale_shift_size / 2] = residual_block0->data[0].beta[n];  //shift
	}

	memory::dims residual_unit9_0_mean_tz = { C };
	memory::dims residual_unit9_0_scale_shift_tz = { 2, C };
	memory::dims residual_unit9_0_src_tz = { N, C, H, W };

	auto residual_unit9_0_bn_mean_md = memory::desc(residual_unit9_0_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto residual_unit9_0_bn_scale_shift_md = memory::desc(residual_unit9_0_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto residual_unit9_0_src_md = memory::desc(
		residual_unit9_0_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto residual_unit9_0_src_mem = residual_unit1_1_user_conv1_dst_mem; // 
	auto residual_unit9_0_mean_mem = memory(residual_unit9_0_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit9_0_bn_mean.data(), residual_unit9_0_mean_mem);
	auto residual_unit9_0_var_mem = memory(residual_unit9_0_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit9_0_bn_var.data(), residual_unit9_0_var_mem);
	auto residual_unit9_0_scale_shift_mem = memory(residual_unit9_0_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(residual_unit9_0_bn_scale_shift.data(), residual_unit9_0_scale_shift_mem);

	flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto residual_unit9_0_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		residual_unit9_0_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);

	auto residual_unit9_0_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(residual_unit9_0_bnrm_fwd_d, attr1, cpu_engine);
	auto residual_unit9_0_bnrm_fwd = batch_normalization_forward(residual_unit9_0_bnrm_fwd_pd);
	residual_unit9_0_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, residual_unit9_0_src_mem },
			{ MKLDNN_ARG_MEAN, residual_unit9_0_mean_mem },
			{ MKLDNN_ARG_VARIANCE, residual_unit9_0_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, residual_unit9_0_scale_shift_mem },
			{ MKLDNN_ARG_DST, residual_unit1_0_bn_dst_mem }
		}
	);

	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 128;
	IC = C, OC = 128, KH = 3, KW = 3;


	memory::dims  residual_unit9_0_conv_src_tz = { N, C, H, W };
	memory::dims  residual_unit9_0_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims  residual_unit9_0_conv_bias_tz = { OC };
	memory::dims  residual_unit9_0_conv_dst_tz = { N, OC, H, W };


	auto  residual_unit9_0_conv_src_md = memory::desc({ residual_unit9_0_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit9_0_conv_bias_md = memory::desc({ residual_unit9_0_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit9_0_conv_weights_md = memory::desc({ residual_unit9_0_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit9_0_conv_dst_md = memory::desc({ residual_unit9_0_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto residual_unit9_0_user_conv1_weights_md = memory::desc(
		residual_unit9_0_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto residual_unit9_0_user_conv1_bias_md = memory::desc({ residual_unit9_0_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	//[Create convolution descriptor]
	auto  residual_unit9_0_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, residual_unit9_0_conv_src_md, residual_unit9_0_conv_weights_md,
		residual_unit9_0_conv_bias_md, residual_unit9_0_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto  residual_unit9_0_conv1_fast_prim_desc = convolution_forward::primitive_desc(residual_unit9_0_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
		reorder(residual_unit1_0_bn_dst_mem, residual_unit1_0_conv1_src_memory)
			.execute(cpu_stream, residual_unit1_0_bn_dst_mem, residual_unit1_0_conv1_src_memory);

	//[Create memory for output]
	// create convolution primitive and add it to net
	auto  residual_unit9_0_fast_conv1 = convolution_forward(residual_unit9_0_conv1_fast_prim_desc);

	residual_unit9_0_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC,  residual_unit1_0_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS,  residual_unit1_0_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, residual_unit1_0_user_conv1_bias_mem },
			{ MKLDNN_ARG_DST,  residual_unit1_0_conv1_dst_memory }
		}
	);


																								   // create reorder between internal and user data if it is needed and
																								   // add it to net after pooling
	if (residual_unit1_0_conv1_dst_memory != residual_unit1_0_user_conv1_dst_mem) {
		reorder(residual_unit1_0_conv1_dst_memory, residual_unit1_0_user_conv1_dst_mem)
			.execute(cpu_stream, residual_unit1_0_conv1_dst_memory, residual_unit1_0_user_conv1_dst_mem);
	}


	/*the residual_unit9_1 residual block*/
	/**********************BN***********************************/
	N = 1, H = hei, W = wid, C = 128;
	int residual_unit9_1_bn_mean_size = C;
	int residual_unit9_1_bn_scale_shift_size = 2 * C;

	std::vector<float> residual_unit9_1_bn_mean(residual_unit9_1_bn_mean_size);
	std::vector<float> residual_unit9_1_bn_var(residual_unit9_1_bn_mean_size);
	std::vector<float> residual_unit9_1_bn_scale_shift(residual_unit9_1_bn_scale_shift_size);

	/*residual_block **/
	for (int n = 0; n < residual_unit9_1_bn_mean_size; n++)
	{
		residual_unit9_1_bn_mean[n] = residual_block0->data[1].u[n];
		residual_unit9_1_bn_var[n] = residual_block0->data[1].std[n];
	}
	for (int n = 0; n < residual_unit9_1_bn_scale_shift_size / 2; n++)
	{
		residual_unit9_1_bn_scale_shift[n] = residual_block0->data[1].alpha[n];    //scale
		residual_unit9_1_bn_scale_shift[n + residual_unit9_1_bn_scale_shift_size / 2] = residual_block0->data[1].beta[n];  //shift
	}

	memory::dims residual_unit9_1_mean_tz = { C };
	memory::dims residual_unit9_1_scale_shift_tz = { 2, C };
	memory::dims residual_unit9_1_src_tz = { N, C, H, W };

	auto residual_unit9_1_bn_mean_md = memory::desc(residual_unit9_1_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto residual_unit9_1_bn_scale_shift_md = memory::desc(residual_unit9_1_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto residual_unit9_1_src_md = memory::desc(
		residual_unit9_1_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto residual_unit9_1_src_mem = residual_unit1_0_user_conv1_dst_mem; // 
	auto residual_unit9_1_mean_mem = memory(residual_unit9_1_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit9_1_bn_mean.data(), residual_unit9_1_mean_mem);
	auto residual_unit9_1_var_mem = memory(residual_unit9_1_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(residual_unit9_1_bn_var.data(), residual_unit9_1_var_mem);
	auto residual_unit9_1_scale_shift_mem = memory(residual_unit9_1_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(residual_unit9_1_bn_scale_shift.data(), residual_unit9_1_scale_shift_mem);

	flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto residual_unit9_1_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		residual_unit9_1_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);

	auto residual_unit9_1_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(residual_unit9_1_bnrm_fwd_d, attr1, cpu_engine);
	auto residual_unit9_1_bnrm_fwd = batch_normalization_forward(residual_unit9_1_bnrm_fwd_pd);
	residual_unit9_1_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, residual_unit9_1_src_mem },
			{ MKLDNN_ARG_MEAN, residual_unit9_1_mean_mem },
			{ MKLDNN_ARG_VARIANCE, residual_unit9_1_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, residual_unit9_1_scale_shift_mem },
			{ MKLDNN_ARG_DST, residual_unit1_1_bn_dst_mem }
		}
	);

	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 128;
	IC = C, OC = 128, KH = 3, KW = 3;


	memory::dims  residual_unit9_1_conv_src_tz = { N, C, H, W };
	memory::dims  residual_unit9_1_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims  residual_unit9_1_conv_bias_tz = { OC };
	memory::dims  residual_unit9_1_conv_dst_tz = { N, OC, H, W };


	auto  residual_unit9_1_conv_src_md = memory::desc({ residual_unit9_1_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit9_1_conv_bias_md = memory::desc({ residual_unit9_1_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit9_1_conv_weights_md = memory::desc({ residual_unit9_1_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  residual_unit9_1_conv_dst_md = memory::desc({ residual_unit9_1_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto residual_unit9_1_user_conv1_weights_md = memory::desc(
		residual_unit9_1_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto residual_unit9_1_user_conv1_bias_md = memory::desc({ residual_unit9_1_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	//[Create convolution descriptor]
	auto  residual_unit9_1_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, residual_unit9_1_conv_src_md, residual_unit9_1_conv_weights_md,
		residual_unit9_1_conv_bias_md, residual_unit9_1_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto  residual_unit9_1_conv1_fast_prim_desc = convolution_forward::primitive_desc(residual_unit9_1_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
		reorder(residual_unit1_1_bn_dst_mem, residual_unit1_1_conv1_src_memory)
			.execute(cpu_stream, residual_unit1_1_bn_dst_mem, residual_unit1_1_conv1_src_memory);

	//[Create memory for output]
	// create convolution primitive and add it to net
	auto  residual_unit9_1_fast_conv1 = convolution_forward(residual_unit9_1_conv1_fast_prim_desc);

	residual_unit9_1_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC,  residual_unit1_1_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS,  residual_unit1_1_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, residual_unit1_1_user_conv1_bias_mem },
			{ MKLDNN_ARG_DST,  residual_unit1_1_conv1_dst_memory }
		}
	);

	//with sum to the first unit output
	write_to_dnnl_memory(first_unit_conv_output_mem.data(), residual_unit1_1_user_conv1_dst_mem);

	auto residual_unit9_src_reorder_pd = reorder::primitive_desc(residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem, attr2);
	reorder(residual_unit9_src_reorder_pd).execute(cpu_stream, residual_unit1_1_conv1_dst_memory, residual_unit1_1_user_conv1_dst_mem);


	/*the last layer*/
	/**********************BN***********************************/
	N = 1, H = hei, W = wid, C = 128;
	int last_unit_bn_mean_size = C;
	int last_unit_bn_scale_shift_size = 2 * C;

	std::vector<float> last_unit_bn_mean(last_unit_bn_mean_size);
	std::vector<float> last_unit_bn_var(last_unit_bn_mean_size);
	std::vector<float> last_unit_bn_scale_shift(last_unit_bn_scale_shift_size);

	/*residual_block **/ residual_block0++;
	for (int n = 0; n < last_unit_bn_mean_size; n++)
	{
		last_unit_bn_mean[n] = sr.the_last_unit.u[n];
		last_unit_bn_var[n] = sr.the_last_unit.std[n];
	}
	for (int n = 0; n < last_unit_bn_scale_shift_size / 2; n++)
	{
		last_unit_bn_scale_shift[n] = sr.the_last_unit.alpha[n];    //scale
		last_unit_bn_scale_shift[n + last_unit_bn_scale_shift_size / 2] = sr.the_last_unit.beta[n];  //shift
	}

	memory::dims last_unit_mean_tz = { C };
	memory::dims last_unit_scale_shift_tz = { 2, C };
	memory::dims last_unit_src_tz = { N, C, H, W };

	auto last_unit_bn_mean_md = memory::desc(last_unit_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto last_unit_bn_scale_shift_md = memory::desc(last_unit_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);
	auto last_unit_src_md = memory::desc(
		last_unit_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case 这里控制memory的layout
	);

	auto last_unit_src_mem = residual_unit1_1_user_conv1_dst_mem; // 
	auto last_unit_mean_mem = memory(last_unit_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(last_unit_bn_mean.data(), last_unit_mean_mem);
	auto last_unit_var_mem = memory(last_unit_bn_mean_md, cpu_engine);
	write_to_dnnl_memory(last_unit_bn_var.data(), last_unit_var_mem);
	auto last_unit_scale_shift_mem = memory(last_unit_bn_scale_shift_md, cpu_engine);
	write_to_dnnl_memory(last_unit_bn_scale_shift.data(), last_unit_scale_shift_mem);
	auto last_unit_bn_dst_mem = memory(last_unit_src_md, cpu_engine);

	flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift | normalization_flags::fuse_norm_relu;

	auto last_unit_bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		last_unit_src_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.000f,     // eps
		flags);

	auto last_unit_bnrm_fwd_pd = batch_normalization_forward::primitive_desc(last_unit_bnrm_fwd_d, attr1, cpu_engine);
	auto last_unit_bnrm_fwd = batch_normalization_forward(last_unit_bnrm_fwd_pd);
	last_unit_bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, last_unit_src_mem },
			{ MKLDNN_ARG_MEAN, last_unit_mean_mem },
			{ MKLDNN_ARG_VARIANCE, last_unit_var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, last_unit_scale_shift_mem },
			{ MKLDNN_ARG_DST, last_unit_bn_dst_mem }
		}
	);

	//{
	//	float * conv3_conv_output = static_cast<float *>(last_unit_bn_dst_mem.get_data_handle());
	//	std::cout << "The last unit 1 BN" << std::endl;
	//	conv3_conv_output = conv3_conv_output ;
	//	// [Check the results]
	//	for (int n = 0; n < 32; n++)
	//	{
	//		std::cout << "i: " << n << " " << std::setw(5) << std::setprecision(6) << *conv3_conv_output;
	//		std::cout << "  ";
	//		conv3_conv_output++;
	//	}
	//	std::cout << std::endl;
	//}
	/*----------------- Conv ---------------------------------*/

	N = 1, H = hei, W = wid, C = 128;
	IC = C, OC = 1, KH = 3, KW = 3;

	int  last_unit_weights_size = OC* IC * KW * KH;
	int  last_unit_bias_size = OC;
	std::vector<float>  last_unit_weights(last_unit_weights_size);
	std::vector<float>  last_unit_bias(last_unit_bias_size);

	for (int offset = 0; offset < last_unit_weights_size; offset++)
	{
		last_unit_weights[offset] = sr.the_last_unit.conv_layer->weights_data[offset];
	}
	//for (int offset = 0; offset < last_unit_weights_size; offset+=9)
	//{
	//	last_unit_weights[offset + 0] = 0;
	//	last_unit_weights[offset + 1] = 0;
	//	last_unit_weights[offset + 2] = 0;
	//	last_unit_weights[offset + 3] = 0;
	//	last_unit_weights[offset + 4] = 0;
	//	last_unit_weights[offset + 5] = 0;
	//	last_unit_weights[offset + 6] = 0;
	//	last_unit_weights[offset + 7] = 0;
	//	last_unit_weights[offset + 8] = 0;
	//}

	for (int offset = 0; offset < last_unit_bias_size; offset++)
	{
		last_unit_bias[offset] = sr.the_last_unit.conv_layer->bias_data[offset];
	}

	memory::dims  last_unit_conv_src_tz = { N, C, H, W };
	memory::dims  last_unit_conv_weights_tz = { OC, IC, KH, KW };
	memory::dims  last_unit_conv_bias_tz = { OC };
	memory::dims  last_unit_conv_dst_tz = { N, OC, H, W };


	auto  last_unit_conv_src_md = memory::desc({ last_unit_conv_src_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  last_unit_conv_bias_md = memory::desc({ last_unit_conv_bias_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  last_unit_conv_weights_md = memory::desc({ last_unit_conv_weights_tz }, memory::data_type::f32, memory::format_tag::any);
	auto  last_unit_conv_dst_md = memory::desc({ last_unit_conv_dst_tz }, memory::data_type::f32, memory::format_tag::any);
	auto last_unit_user_conv1_weights_md = memory::desc(
		last_unit_conv_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto last_unit_user_conv1_bias_md = memory::desc({ last_unit_conv_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	auto last_unit_user_conv1_weights_mem = memory(last_unit_user_conv1_weights_md, cpu_engine);
	write_to_dnnl_memory(last_unit_weights.data(), last_unit_user_conv1_weights_mem);
	auto last_unit_user_conv1_bias_mem = memory(last_unit_user_conv1_bias_md, cpu_engine);
	write_to_dnnl_memory(last_unit_bias.data(), last_unit_user_conv1_bias_mem);

	//[Create convolution descriptor]
	auto  last_unit_conv1_fast_desc = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, last_unit_conv_src_md, last_unit_conv_weights_md,
		last_unit_conv_bias_md, last_unit_conv_dst_md, conv_strides, conv_padding,
		conv_padding);
	//[Create convolution descriptor]

	//[Create convolution primitive descriptor]
	auto  last_unit_conv1_fast_prim_desc = convolution_forward::primitive_desc(last_unit_conv1_fast_desc, cpu_engine);
	//[Create convolution primitive descriptor]

	//[Reorder data and weights]
	auto  last_unit_conv1_src_memory = last_unit_bn_dst_mem;
	if (last_unit_conv1_fast_prim_desc.src_desc() != last_unit_bn_dst_mem.get_desc()) {
		last_unit_conv1_src_memory = memory(last_unit_conv1_fast_prim_desc.src_desc(), cpu_engine);
		reorder(last_unit_bn_dst_mem, last_unit_conv1_src_memory)
			.execute(cpu_stream, last_unit_bn_dst_mem, last_unit_conv1_src_memory);
	}
	auto  last_unit_conv1_weights_memory = last_unit_user_conv1_weights_mem;
	if (last_unit_conv1_fast_prim_desc.weights_desc() != last_unit_user_conv1_weights_mem.get_desc()) {
		last_unit_conv1_weights_memory = memory(last_unit_conv1_fast_prim_desc.weights_desc(), cpu_engine);
		reorder(last_unit_user_conv1_weights_mem, last_unit_conv1_weights_memory)
			.execute(cpu_stream, last_unit_user_conv1_weights_mem, last_unit_conv1_weights_memory);
	}

	//[Create memory for output]
	auto  last_unit_conv1_dst_memory = memory(last_unit_conv1_fast_prim_desc.dst_desc(), cpu_engine);
	//[Create memory for output]
	// create convolution primitive and add it to net
	auto  last_unit_fast_conv1 = convolution_forward(last_unit_conv1_fast_prim_desc);

	last_unit_fast_conv1.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC,  last_unit_conv1_src_memory },
			{ MKLDNN_ARG_WEIGHTS,  last_unit_conv1_weights_memory },
			{ MKLDNN_ARG_BIAS, last_unit_user_conv1_bias_mem },
			{ MKLDNN_ARG_DST,  last_unit_conv1_dst_memory }
		}
	);

	//with sum to the first unit output
	//auto  last_unit_user_conv1_dst_mem = memory(first_unit_user_dst_md, cpu_engine);
	//write_to_dnnl_memory(first_unit_conv_output_mem.data(), last_unit_user_conv1_dst_mem);
	auto last_unit_user_dst_md = memory::desc(
		last_unit_conv_dst_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case
	);
	auto last_unit_user_conv_dst_mem = memory(last_unit_user_dst_md, cpu_engine);  //for conv output
	write_to_dnnl_memory(image.data(), last_unit_user_conv_dst_mem);

	auto last_unit_src_reorder_pd = reorder::primitive_desc(last_unit_conv1_dst_memory, last_unit_user_conv_dst_mem, attr2);
	reorder(last_unit_src_reorder_pd).execute(cpu_stream, last_unit_conv1_dst_memory, last_unit_user_conv_dst_mem);

	cpu_stream.wait();


	read_from_dnnl_memory(image.data(), last_unit_user_conv_dst_mem);

	ConvLayer hR1(wid, hei);
	hR1.data = new float[wid * hei];
	memcpy(hR1.data, image.data(), wid*hei * sizeof(float));

	//{
	//	float * conv3_conv_output = static_cast<float *>(last_unit_user_conv_dst_mem.get_data_handle());
	//	std::cout << "The last unit 1 Conv" << std::endl;
	//	conv3_conv_output = conv3_conv_output;
	//	// [Check the results]
	//	for (int n = 0; n < 32 ; n++)
	//	{
	//		std::cout << "i: " << n << " " << std::setw(5) << std::setprecision(6) << *conv3_conv_output;
	//		std::cout << "  ";
	//		conv3_conv_output++;
	//	}
	//	std::cout << std::endl;
	//}

#endif 
	/****************************************************************************/
	/* temp define */
#if 0
	auto user_dst3_md = memory::desc(
		first_unit_conv_dst_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nchw    // memory format, NHWC in this case
	);
	auto user_conv3_dst1_mem = memory(user_dst3_md, cpu_engine);  //for conv output

	// create reorder between internal and user data if it is needed and
	// add it to net after pooling
	if (first_unit_conv1_dst_memory != user_conv3_dst1_mem) {
		reorder(first_unit_conv1_dst_memory, user_conv3_dst1_mem)
			.execute(cpu_stream, first_unit_conv1_dst_memory, user_conv3_dst1_mem);
	}

	/* end of temp define */


	cpu_stream.wait();

	float *conv3_conv_output = static_cast<float *>(user_conv3_dst1_mem.get_data_handle());
	std::cout << "The 1st Unit Conv" << std::endl;
	conv3_conv_output = conv3_conv_output + wid*hei + wid * 5 + 5;
	// [Check the results]
	for (int n = 0; n < W*2; n++)
	{
					std::cout <<"i: "<<n << " "<< std::setw(5) << std::setprecision(6) << *conv3_conv_output;
						std::cout << "  ";
					conv3_conv_output++;
	}
	std::cout << std::endl;
#endif
	/**********************************************************/


#if 0
	 //尾单位层  ---->开始
	cout << "The last layer... " << endl;

	vl_BatchNorm(Source, sr.the_last_unit.u, sr.the_last_unit.std);//函数

	vl_Scale(Source, sr.the_last_unit.alpha, sr.the_last_unit.beta);//函数  

	vl_nnrelu(Source);


	//save_mat ("dd2.txt",源->data,源->width,源->height,源->depth) ; //保存
	delete[]convfea1.data;  convfea1.data = NULL;
	delete[]Target->data;  Target->data = NULL;

	CurrentLayer = sr.the_last_unit.conv_layer;

	// 3倍重建图
	ConvLayer hR1(wid, hei);

	hR1.data = new float[wid * hei];

	vl_nnconv(Source, &hR1, CurrentLayer, 1, 1, 1, 1, 1, 1);


	//尾单位层  <----结束
	delete[]Source->data;  Source->data = NULL;


	//求和
	cout << "Combine output image... " << endl;

	add_matrix(&im_b, &hR1);

	//cv::Mat im_tt2;
	//im_tt2 = Matrix2Im(&hR1);
	//cv::imshow("gray", im_tt2);

#endif
	delete[]im_b.data;  im_b.data = NULL;

	//save_卷积层2jpg(&hR1,"hR1");
	//YUV2RGB(&hR1, &U, &V, &jpg);
	Matrix2ImBGR(&hR1, &U, &V, &jpg);

	char txt[255];
	sprintf(txt, "DRRN_%dx_rebuilt.jpg", up_scale);

	//savejpg(&jpg, txt);//完成图
	cv::imwrite(txt, jpg);
	imshow(txt, jpg);

}


int main()
{


	//载入图片
	char jpgname[] = "c:\\work\\fsrcnn\\dreamstime_xxl_123399800.jpg";//lena.jpg
	loadjpg(jpgname);

	char txt[255];
	//显示原图
	//putimage(0, 0, &jpg);
	sprintf(txt, "original image: %dx%d", jpg.cols,jpg.rows);
	imshow(txt, jpg);

	// 放大倍率 2，3 或 4 倍
	int up_scale = 3;

	int height = jpg.rows;
	int width = jpg.cols;

	cout << width << endl;
	cout << height << endl;


	// DRRN 高分辨率重建
	clock_t start_t, end_t;//计算时间
	start_t = clock();
	DRRN(up_scale);

	end_t = clock();

	double total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
	cout << "total cost time: " << total_t << endl;


	while (1)
	{
		if (cv::waitKey(30) == 27 /*ESC*/)
		{
			break;
		}
	}




	//closegraph();
	//system("pause");
	return 0;
}
