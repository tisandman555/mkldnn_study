// mkldnntest1.cpp : Defines the entry point for the console application.
//


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
	/*const int N = 1, H = 5, W = 5, C = 2;*/
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
	//const int weights_size = OC* IC * KW * KH;
	//const int bias_size = OC;
	const int bn_mean_size = C;
	const int bn_scale_shift_size = 2 * C;

	// Allocate a buffer for the image
	std::vector<float> image(image_size);
	std::vector<float> mean(bn_mean_size);
	std::vector<float> var(bn_mean_size);
	std::vector<float> scale_shift(bn_scale_shift_size);

	// Initialize the image with some values
	for (int n = 0; n < N; n++)
		for (int h = 0; h < H; h++)
			for (int w = 0; w < W; w++)
				for (int c = 0; c < C; c++) {
					int off = offset(n, h, w, c); // Get the physical offset of a pixel
												  /*image[off] = -std::cos(off / 10.f);*/
												  /*image[off] = n * 1000 + h * 100 + w * 10 + c;*/
					image[off] = off;
					//image[off] = -std::cos(off / 10.f);
					//std::cout << "off=" << off << " : " << image[off] << std::endl;
				}
	for (int n = 0; n < bn_mean_size; n++)
	{
		mean[n] = 1.0;
		var[n] = 4.0;
		if (n >= bn_mean_size / 2)
		{
			mean[n] = 0.0;
			var[n] = 4.0;
		}
	}
	for (int n = 0; n < bn_scale_shift_size; n++)
	{
		scale_shift[n] = 2;    //scale
		if (n >= bn_scale_shift_size / 2)
		{
			scale_shift[n] = -1;    //shift
		}
	}



#ifdef DISPLAY

	// output input matrix
	for (int n = 0; n < N; n++)
	{
		for (int h = 0; h < H; h++)
		{
			for (int w = 0; w < W; w++)
			{
				for (int c = 0; c < C; c++) {
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

#endif

	memory::dims conv3_src_tz = { N, C, H, W };
	memory::dims conv3_mean_tz = { C };
	memory::dims conv3_scale_shift_tz = { 2, C };

	// [Init src_md]
	auto user_src3_md = memory::desc(
		conv3_src_tz, // logical dims, the order is defined by a primitive
		memory::data_type::f32,     // tensor's data type
		memory::format_tag::nhwc    // memory format, NHWC in this case 这里控制memory的layout
	);

	// create user memory
	auto user_conv3_src_mem = memory(user_src3_md, cpu_engine, image.data());

	/*********************** Batch Normal ***************************************************/

	auto conv3_mean_md = memory::desc(conv3_mean_tz, memory::data_type::f32, memory::format_tag::x);
	auto conv3_scale_shift_md = memory::desc(conv3_scale_shift_tz, memory::data_type::f32, memory::format_tag::nc);

	auto mean_mem = memory(conv3_mean_md, cpu_engine, mean.data());
	auto var_mem = memory(conv3_mean_md, cpu_engine, var.data());
	auto scale_shift_mem = memory(conv3_scale_shift_md, cpu_engine, scale_shift.data());

	auto user_bn_dst_mem = memory(user_src3_md, cpu_engine);

	//normalization_flags flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift;// | normalization_flags::fuse_norm_relu;
	// set flags for different flavors (use | to combine flags)
	//      use_global_stats -- do not compute mean and variance in the primitive, user has to provide them
	//      use_scale_shift  -- in addition to batch norm also scale and shift the result

	normalization_flags flags = normalization_flags::use_global_stats | normalization_flags::use_scale_shift;// | normalization_flags::fuse_norm_relu;
	auto bnrm_fwd_d = batch_normalization_forward::desc(
		prop_kind::forward_inference, // might be forward_inference, backward, backward_data
		user_src3_md,  // data descriptor (i.e. sizes, data type, and layout)
		0.001f,     // eps
		flags);

#if 1
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
#endif

	auto bnrm_fwd_pd = batch_normalization_forward::primitive_desc(bnrm_fwd_d, attr1, cpu_engine);
	//auto bnrm_fwd_pd = batch_normalization_forward::primitive_desc(bnrm_fwd_d, cpu_engine);
	auto bnrm_fwd = batch_normalization_forward(bnrm_fwd_pd);
	bnrm_fwd.execute(
		cpu_stream,
		{
			{ MKLDNN_ARG_SRC, user_conv3_src_mem },
			{ MKLDNN_ARG_MEAN, mean_mem },
			{ MKLDNN_ARG_VARIANCE, var_mem },
			{ MKLDNN_ARG_SCALE_SHIFT, scale_shift_mem },
			{ MKLDNN_ARG_DST, user_bn_dst_mem }
		}
	);

	/**************************************************************************/
	// Wait the stream to complete the execution
	cpu_stream.wait();


#ifdef DISPLAY

	float *conv3_bn_output = static_cast<float *>(user_bn_dst_mem.get_data_handle());
	
	std::cout << "Batch Normal" << std::endl;
	// [Check the results]
	for (int n = 0; n < N; ++n)
	{
		for (int h = 0; h < H; ++h)
		{
			for (int w = 0; w < W; ++w)
			{
				for (int c = 0; c < C; ++c) {
					std::cout << std::setfill(' ') << std::setw(5) << std::setprecision(6) << *conv3_bn_output++ << " ";
				}
				std::cout << "|";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}



#endif


}

using namespace std;

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

	std::cout << "Example passes" << std::endl;
	return 0;
}
// [Main]
