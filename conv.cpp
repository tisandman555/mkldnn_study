// mkldnntest1.cpp : Defines the entry point for the console application.
// 测试MKLDNN里conv的执行，所有层都采用自定义的格式，不做格式转换，导致无法用jit avx2的函数加速，性能不好


#include <iostream>
#include <sstream>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include <iomanip>

// [Prologue]
#include "mkldnn.hpp"

// Optional header to access debug functions like `mkldnn_status2str()`
#include "mkldnn_debug.h"

using namespace mkldnn;
// [Prologue]
#define DISPLAY

void cpu_test_conv() {
	// [Initialize engine]
	engine cpu_engine(engine::kind::cpu, 0);
	// [Initialize engine]

	// [Initialize stream]
	stream cpu_stream(cpu_engine);
	// [Initialize stream]

	// [Create user's data]
	const int N = 1, H = 5, W = 5, C = 2;
	const int IC = C, OC = IC, KH = 3, KW = 3;

	// Compute physical strides for each dimension
	const int stride_N = H * W * C;
	const int stride_H = W * C;
	const int stride_W = C;
	const int stride_C = 1;

	const int stride_OC = KH * KW * IC;
	const int stride_IC = KW * KH;
	const int stride_KH = KW;
	const int stride_KW = 1;


	// An auxiliary function that maps logical index to the physical offset
	auto offset = [=](int n, int h, int w, int c)
	{ return n * stride_N + h * stride_H + w * stride_W + c * stride_C; };

	auto offset_ws = [=](int n, int c, int h, int w)
	{ return n * stride_OC + h * stride_KH + w * stride_KW + c * stride_IC; };


	// The image size
	const int image_size = N * H * W * C;
	const int weights_size = OC* IC * KW * KH;
	const int bias_size = OC;

	// Allocate a buffer for the image
	std::vector<float> image(image_size);
	std::vector<float> weights(weights_size);
	std::vector<float> bias(bias_size);

	// Initialize the image with some values
	for (int n = 0; n < N; ++n)
		for (int h = 0; h < H; ++h)
			for (int w = 0; w < W; ++w)
				for (int c = 0; c < C; ++c) {
					int off = offset(n, h, w, c); // Get the physical offset of a pixel
												  /*image[off] = -std::cos(off / 10.f);*/
												  /*image[off] = n * 1000 + h * 100 + w * 10 + c;*/
					image[off] = off;
					//std::cout << "off=" << off << " : " << image[off] << std::endl;
				}
	// [Create user's data]
	for (int n = 0; n < OC; ++n)
	{
		for (int c = 0; c < IC; ++c)
		{
			for (int h = 0; h < KH; ++h)
			{
				for (int w = 0; w < KW; ++w)
				{
					int off = offset_ws(n, c, h, w); // Get the physical offset of a pixel
													 //weights[off] = -std::cos(off / 10.f);
					if (n == 0)
					{
						weights[off] = 1;
					}
					else
					{
						weights[off] = 2;
					}
				}
			}
		}
	}
	for (int n = 0; n < OC; ++n)
	{
		bias[n] = 0.01+0.01*n;
	}



#ifdef DISPLAY

	// output input matrix
	for (int n = 0; n < N; ++n)
	{
		for (int h = 0; h < H; ++h)
		{
			for (int w = 0; w < W; ++w)
			{
				for (int c = 0; c < C; ++c) {
					int off = offset(n, h, w, c); // Get the physical offset of a pixel
					std::cout << std::setfill(' ') << std::setw(5) << image[off] << " ";
					//std::cout << "off=" << off << " : " << image[off] << std::endl;
				}
				std::cout << "|";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
	std::cout << std::endl;

	for (int n = 0; n < OC; ++n)
	{
		for (int c = 0; c < IC; ++c)
		{
			for (int h = 0; h < KH; ++h)
			{
				for (int w = 0; w < KW; ++w)
				{
					int off = offset_ws(n, c, h, w); // Get the physical offset of a pixel
					std::cout << std::setfill(' ') << std::setw(5) << weights[off] << " ";
					//std::cout << "off=" << off << " : " << image[off] << std::endl;
				}
				std::cout << "|" << std::endl;
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
#endif

	memory::dims conv3_src_tz = { N, C, H, W };
	memory::dims conv3_weights_tz = { OC, IC, KH, KW };
	memory::dims conv3_bias_tz = { OC };
	memory::dims conv3_dst_tz = { N, OC, H, W };
	memory::dims conv3_strides = { 1, 1 };
	memory::dims conv3_padding = { 1, 1 };

	// [Init src_md]
	auto src3_md = memory::desc(
		conv3_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nhwc    // memory format, NHWC in this case
	);

	// [Init src_md]
	auto conv3_weights_md = memory::desc(
		conv3_weights_tz, memory::data_type::f32,
		memory::format_tag::oihw // 
	);
	auto conv3_bias_md = memory::desc({ conv3_bias_tz }, memory::data_type::f32, memory::format_tag::x);

	auto dst3_md = memory::desc(
		conv3_dst_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nhwc    // memory format, NHWC in this case
	);

	auto user_src_mem = memory(src3_md, cpu_engine, image.data());
	auto user_conv3_weights_mem = memory(conv3_weights_md, cpu_engine, weights.data());
	auto user_conv3_bias_mem = memory(conv3_bias_md, cpu_engine, bias.data());
	// For dst_mem the library allocates buffer
	auto user_conv3_dst_mem = memory(dst3_md, cpu_engine);  //for conv output

	auto conv3_d = convolution_forward::desc(prop_kind::forward_inference,
		algorithm::convolution_direct, src3_md, conv3_weights_md,
		conv3_bias_md,
		dst3_md, conv3_strides, conv3_padding,
		conv3_padding);
	auto conv3_pd = convolution_forward::primitive_desc(conv3_d, cpu_engine);

	// create convolution primitive and add it to net
	auto conv3 = convolution_forward(conv3_pd);

	conv3.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, user_src_mem },
			{ MKLDNN_ARG_WEIGHTS, user_conv3_weights_mem },
			{ MKLDNN_ARG_BIAS, user_conv3_bias_mem },
			{ MKLDNN_ARG_DST, user_conv3_dst_mem }
		}
	);


	// Wait the stream to complete the execution
	cpu_stream.wait();

	float *conv3_output = static_cast<float *>(user_conv3_dst_mem.get_data_handle());

#ifdef DISPLAY
	// [Check the results]
	for (int n = 0; n < N; ++n)
	{
		for (int h = 0; h < H; ++h)
		{
			for (int w = 0; w < W; ++w)
			{
				for (int c = 0; c < OC; ++c)
				{
					std::cout << std::setfill(' ') << std::setw(5) << std::setprecision(6) << *conv3_output++ << " ";
				}
				std::cout << "|" ;
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
#endif
}
// [Main]
int main(int argc, char **argv) {
	try {
		cpu_test_conv();
	}
	catch (mkldnn::error &e) {
		std::cerr << "Intel MKL-DNN error: " << e.what() << std::endl
			<< "Error status: " << mkldnn_status2str(e.status) << std::endl;
		return 1;
	}
	catch (std::string &e) {
		std::cerr << "Error in the example: " << e << std::endl;
		return 2;
	}

	//system("pause");
	return 0;
}
// [Main]
