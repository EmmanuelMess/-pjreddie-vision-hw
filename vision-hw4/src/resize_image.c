#include <math.h>
#include "image.h"

float nn_interpolate(image im, float x, float y, int c)
{
	int oldx = (int) roundf(x);
	int oldy = (int) roundf(y);

	return get_pixel(im, oldx, oldy, c);
}

image nn_resize(image im, int w, int h)
{
    image resized = make_image(w, h, im.c);

    float rx = (float)(im.w)/(float)(w);
    float ry = (float)(im.h)/(float)(h);


	for (int i = 0; i < resized.w; ++i) {
		for (int j = 0; j < resized.h; ++j) {
			for (int k = 0; k < resized.c; ++k) {
				float result = nn_interpolate(im, (i + 0.5f)*rx - 0.5f, (j + 0.5f)*ry - 0.5f, k);
				set_pixel(resized, i, j, k, result);
			}
		}
	}

	return resized;
}

float lerp(float a, float b, float alpha) {
	return a + (b-a) * alpha;
}

float blerp(float top_left, float top_right, float bottom_right, float bottom_left, float tx, float ty) {
	float lerpxtop = lerp(top_left, top_right, tx);
	float lerpxbottom = lerp(bottom_left, bottom_right, tx);

	float value = lerp(lerpxtop, lerpxbottom, ty);

	return value;
}

float bilinear_interpolate(image im, float x, float y, int c)
{
	int left = (int) floorf(x);
	int top = (int) floorf(y);
	int right = (int) left + 1;
	int bottom = (int) top + 1;

	float top_left = get_pixel(im, left, top, c);
	float top_right = get_pixel(im, right, top, c);
	float bottom_right = get_pixel(im, right, bottom, c);
	float bottom_left = get_pixel(im, left, bottom, c);

    return blerp(top_left, top_right, bottom_right, bottom_left, x - left, y - top);
}

image bilinear_resize(image im, int w, int h)
{
	image resized = make_image(w, h, im.c);


	float rx = (float)(im.w)/(float)(w);
	float ry = (float)(im.h)/(float)(h);


	for (int i = 0; i < resized.w; ++i) {
		for (int j = 0; j < resized.h; ++j) {
			for (int k = 0; k < resized.c; ++k) {
				float result = bilinear_interpolate(im, (i+0.5f)*rx - 0.5f, (j+0.5f)*ry - 0.5f, k);
				set_pixel(resized, i, j, k, result);
			}
		}
	}

	return resized;
}

