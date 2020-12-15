#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <setjmp.h>
#include "image.h"
#define TWOPI 6.2831853

void l1_normalize(image im)
{
	float constant = 1.0f/(im.w*im.h);

	for (int i = 0; i < im.w; ++i) {
		for (int j = 0; j < im.h; ++j) {
			for (int k = 0; k < im.c; ++k) {
				set_pixel(im, i, j, k, get_pixel(im, i, j, k) * constant);
			}
		}
	}
}

image make_box_filter(int w)
{
	image im = make_image(w, w,1);
	for (int i = 0; i < im.w; ++i) {
		for (int j = 0; j < im.h; ++j) {
			set_pixel(im, i, j, 0, 1);
		}
	}
	l1_normalize(im);
	return im;
}

image convolve_image(image im, image filter, int preserve)
{
	if (filter.c == 1 && im.c > 1) {
		if(preserve) {
			image result = make_image(im.w, im.h, im.c);

#pragma omp parallel for
			for (int i = 0; i < im.w; ++i) {
#pragma omp parallel for
				for (int j = 0; j < im.h; ++j) {
					for (int k = 0; k < im.c; ++k) {
						float sum = 0;

						for (int l = 0; l < filter.w; ++l) {
							for (int m = 0; m < filter.h; ++m) {
								sum += get_pixel(im, i + l  - (filter.w - 1.0f)/2.0f, j + m - (filter.h - 1.0f)/2.0f, k) * get_pixel(filter, l, m, 0);
							}
						}

						set_pixel(result, i, j, k, sum);
					}
				}
			}

			return result;
		} else {
			image result = make_image(im.w, im.h, 1);

#pragma omp parallel for
			for (int i = 0; i < im.w; ++i) {
#pragma omp parallel for
				for (int j = 0; j < im.h; ++j) {
					float sum = 0;
					for (int k = 0; k < im.c; ++k) {
						for (int l = 0; l < filter.w; ++l) {
							for (int m = 0; m < filter.h; ++m) {
								sum += get_pixel(im, i + l - (filter.w - 1.0f)/2.0f, j + m - (filter.h - 1.0f)/2.0f, k) * get_pixel(filter, l, m, 0);
							}
						}
					}
					set_pixel(result, i, j, 0, sum);
				}
			}

			return result;
		}
	} else if (filter.c == im.c) {
		if(preserve) {
			image result = make_image(im.w, im.h, im.c);

#pragma omp parallel for
			for (int i = 0; i < im.w; ++i) {
#pragma omp parallel for
				for (int j = 0; j < im.h; ++j) {
					for (int k = 0; k < im.c; ++k) {
						float sum = 0;

						for (int l = 0; l < filter.w; ++l) {
							for (int m = 0; m < filter.h; ++m) {
								sum += get_pixel(im, i + l - (filter.w - 1.0f)/2.0f, j + m - (filter.h - 1.0f)/2.0f, k) * get_pixel(filter, l, m, k);
							}
						}

						set_pixel(result, i, j, k, sum);
					}
				}
			}

			return result;
		} else {
			image result = make_image(im.w, im.h, 1);

#pragma omp parallel for
			for (int i = 0; i < im.w; ++i) {
#pragma omp parallel for
				for (int j = 0; j < im.h; ++j) {
					float sum = 0;

					for (int k = 0; k < im.c; ++k) {
						for (int l = 0; l < filter.w; ++l) {
							for (int m = 0; m < filter.h; ++m) {
								sum += get_pixel(im, i + l - (filter.w - 1.0f)/2.0f, j + m - (filter.h - 1.0f)/2.0f, k) * get_pixel(filter, l, m, k);
							}
						}
					}

					set_pixel(result, i, j, 0, sum);
				}
			}

			return result;
		}
	}

	return make_image(1,1,1);
}

image make_highpass_filter()
{
	image result = make_image(3, 3, 1);
	set_pixel(result, 0, 0, 0, 0);
	set_pixel(result, 1, 0, 0, -1);
	set_pixel(result, 2, 0, 0, 0);
	set_pixel(result, 0, 1, 0, -1);
	set_pixel(result, 1, 1, 0, 4);
	set_pixel(result, 2, 1, 0, -1);
	set_pixel(result, 0, 2, 0, 0);
	set_pixel(result, 1, 2, 0, -1);
	set_pixel(result, 2, 2, 0, 0);
	return result;
}

image make_sharpen_filter()
{
	image result = make_image(3, 3, 1);
	set_pixel(result, 0, 0, 0, 0);
	set_pixel(result, 1, 0, 0, -1);
	set_pixel(result, 2, 0, 0, 0);
	set_pixel(result, 0, 1, 0, -1);
	set_pixel(result, 1, 1, 0, 5);
	set_pixel(result, 2, 1, 0, -1);
	set_pixel(result, 0, 2, 0, 0);
	set_pixel(result, 1, 2, 0, -1);
	set_pixel(result, 2, 2, 0, 0);
	return result;
}

image make_emboss_filter()
{
	image result = make_image(3, 3, 1);
	set_pixel(result, 0, 0, 0, -2);
	set_pixel(result, 1, 0, 0, -1);
	set_pixel(result, 2, 0, 0, 0);
	set_pixel(result, 0, 1, 0, -1);
	set_pixel(result, 1, 1, 0, 1);
	set_pixel(result, 2, 1, 0, 1);
	set_pixel(result, 0, 2, 0, 0);
	set_pixel(result, 1, 2, 0, 1);
	set_pixel(result, 2, 2, 0, 2);
	return result;
}

// Question 2.2.1: Which of these filters should we use preserve when we run our convolution and which ones should we not? Why?
// Answer: Highpass should not preserve, it needs to detect edges _across_ colors, not edges on each color.
//         Sharpen should preserve, it should sharpen each color independently.
//         Emboss should preserve, visually, it seems to emboss each color independently.

// Question 2.2.2: Do we have to do any post-processing for the above filters? Which ones and why?
// Answer: TODO

image make_gaussian_filter(float sigma)
{
	int size = (int) ceil(6*sigma) % 2 == 0 ? ceil(6*sigma) + 1 : ceil(6*sigma);
	image result = make_image(size,size,1);
	float k = 1.0f/(TWOPI*sigma*sigma);

	for (int i = 0; i < size; ++i) {
		for (int j = 0; j < size; ++j) {
			int x = i - (size - 1.0f)/2.0f;
			int y = j - (size - 1.0f)/2.0f;
			float z = -1.0f*(x*x + y*y)/(2 * sigma * sigma);
			set_pixel(result, i, j, 0, k * expf(z));
		}
	}

	//l1_normalize(result);

	return result;
}

image add_image(image a, image b)
{
	assert(a.w == b.w && a.h == b.h && a.c == b.c);
	image result = make_image(a.w, a.h, a.c);
	for (int i = 0; i < a.w; ++i) {
		for (int j = 0; j < a.h; ++j) {
			for (int k = 0; k < a.c; ++k) {
				set_pixel(result, i, j, k, get_pixel(a, i, j, k) + get_pixel(b, i, j, k));
			}
		}
	}
	return result;
}

image sub_image(image a, image b)
{
	assert(a.w == b.w && a.h == b.h && a.c == b.c);
	image result = make_image(a.w, a.h, a.c);
	for (int i = 0; i < a.w; ++i) {
		for (int j = 0; j < a.h; ++j) {
			for (int k = 0; k < a.c; ++k) {
				set_pixel(result, i, j, k, get_pixel(a, i, j, k) - get_pixel(b, i, j, k));
			}
		}
	}
	return result;
}

image make_gx_filter()
{
	image result = make_image(3, 3, 1);
	set_pixel(result, 0, 0, 0, -1);
	set_pixel(result, 1, 0, 0, 0);
	set_pixel(result, 2, 0, 0, 1);
	set_pixel(result, 0, 1, 0, -2);
	set_pixel(result, 1, 1, 0, 0);
	set_pixel(result, 2, 1, 0, 2);
	set_pixel(result, 0, 2, 0, -1);
	set_pixel(result, 1, 2, 0, 0);
	set_pixel(result, 2, 2, 0, 1);
	return result;
}

image make_gy_filter()
{
	image result = make_image(3, 3, 1);
	set_pixel(result, 0, 0, 0, -1);
	set_pixel(result, 1, 0, 0, -2);
	set_pixel(result, 2, 0, 0, -1);
	set_pixel(result, 0, 1, 0, 0);
	set_pixel(result, 1, 1, 0, 0);
	set_pixel(result, 2, 1, 0, 0);
	set_pixel(result, 0, 2, 0, 1);
	set_pixel(result, 1, 2, 0, 2);
	set_pixel(result, 2, 2, 0, 1);
	return result;
}

void feature_normalize(image im)
{
	float min = INFINITY;
	float max = -INFINITY;

	for (int i = 0; i < im.w; ++i) {
		for (int j = 0; j < im.h; ++j) {
			for (int k = 0; k < im.c; ++k) {
				float v = get_pixel(im, i, j, k);
				if (v < min) {
					min = v;
				}
				if(v > max) {
					max = v;
				}
			}
		}
	}

	float normalizator = max - min;

	for (int i = 0; i < im.w; ++i) {
		for (int j = 0; j < im.h; ++j) {
			for (int k = 0; k < im.c; ++k) {
				if (normalizator == 0) {
					set_pixel(im, i, j, k, 0);
				} else {
					float v = get_pixel(im, i, j, k);
					set_pixel(im, i, j, k, (v - min)/normalizator);
				}
			}
		}
	}
}

image mult_image(image a, image b)
{
	assert(a.w == b.w && a.h == b.h && a.c == b.c);
	image result = make_image(a.w, a.h, a.c);

#pragma omp parallel for
	for (int i = 0; i < a.w; ++i) {
#pragma omp parallel for
		for (int j = 0; j < a.h; ++j) {
			for (int k = 0; k < a.c; ++k) {
				set_pixel(result, i, j, k, get_pixel(a, i, j, k) * get_pixel(b, i, j, k));
			}
		}
	}
	return result;
}

image div_image(image a, image b)
{
	assert(a.w == b.w && a.h == b.h && a.c == b.c);
	image result = make_image(a.w, a.h, a.c);
	for (int i = 0; i < a.w; ++i) {
		for (int j = 0; j < a.h; ++j) {
			for (int k = 0; k < a.c; ++k) {
				set_pixel(result, i, j, k, get_pixel(a, i, j, k) / get_pixel(b, i, j, k));
			}
		}
	}
	return result;
}

image sqrt_image(image a)
{
	image result = make_image(a.w, a.h, a.c);
	for (int i = 0; i < a.w; ++i) {
		for (int j = 0; j < a.h; ++j) {
			for (int k = 0; k < a.c; ++k) {
				set_pixel(result, i, j, k, sqrtf(get_pixel(a, i, j, k)));
			}
		}
	}
	return result;
}

image atan2_image(image y, image x)
{
	image result = make_image(y.w, y.h, y.c);
	for (int i = 0; i < y.w; ++i) {
		for (int j = 0; j < y.h; ++j) {
			for (int k = 0; k < y.c; ++k) {
				set_pixel(result, i, j, k, atan2f(get_pixel(y, i, j, k), get_pixel(x, i, j, k)));
			}
		}
	}
	return result;
}

image *sobel_image(image im)
{
	image * result = calloc(2, sizeof(image));

	image gx_filter = make_gx_filter();
	image gy_filter = make_gy_filter();

	image gx = convolve_image(im, gx_filter, 0);
	image gy = convolve_image(im, gy_filter, 0);

	image gx2 = mult_image(gx, gx);
	image gy2 = mult_image(gy, gy);

	image sum = add_image(gx2, gy2);
	image magnitude = sqrt_image(sum);

	feature_normalize(magnitude);

	result[0] = magnitude;

	image theta = atan2_image(gy, gx);

	feature_normalize(theta);

	result[1] = theta;

	free_image(gx);
	free_image(gy);
	free_image(gx2);
	free_image(gy2);
	free_image(sum);

	return result;
}

image colorize_sobel(image im)
{
	// TODO
	return make_image(1,1,1);
}
