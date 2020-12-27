#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "image.h"

float get_pixel(image im, int x, int y, int c)
{
	assert(0 <= c && c < im.c);

	x = x < 0? 0:x;
	y = y < 0? 0:y;
	c = c < 0? 0:c;

	x = x >= im.w? im.w - 1 : x;
	y = y >= im.h? im.h - 1 : y;
	c = c >= im.c? im.c - 1 : c;

    return im.data[x + im.w*y + im.w*im.h*c];
}

void set_pixel(image im, int x, int y, int c, float v)
{
	if (x < 0 || y < 0 || c < 0
			|| x >= im.w || y >= im.h || c >= im.c) return;

	im.data[x + im.w*y + im.w*im.h*c] = v;
}

image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
    memcpy(copy.data, im.data, im.w*im.h*im.c*sizeof(float));
    return copy;
}

image rgb_to_grayscale(image im)
{
    assert(im.c == 3);
    image gray = make_image(im.w, im.h, 1);
	for (int i = 0; i < im.w; ++i) {
		for (int j = 0; j < im.h; ++j) {
			set_pixel(gray, i, j, 0,
			 0.299 * get_pixel(im, i, j, 0) + 0.587 * get_pixel(im, i, j, 1) + .114  * get_pixel(im, i, j, 2));
		}
	}

    return gray;
}

void shift_image(image im, int c, float v)
{
	for (int i = 0; i < im.w; ++i) {
		for (int j = 0; j < im.h; ++j) {
			set_pixel(im, i, j, c, get_pixel(im, i, j, c) + v);
		}
	}
}

void clamp_image(image im)
{
	for (int i = 0; i < im.w; ++i) {
		for (int j = 0; j < im.h; ++j) {
			for (int k = 0; k < im.c; ++k) {
				float value = get_pixel(im, i, j, k);
				value = value < 0? 0: value;
				value = value > 1? 1: value;
				set_pixel(im, i, j, k, value);
			}
		}
	}
}


// These might be handy
float three_way_max(float a, float b, float c)
{
    return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
    return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

void rgb_to_hsv(image im)
{
	for(int i = 0; i < im.w; i++) {
		for (int j = 0; j < im.h; ++j) {
			float r = get_pixel(im, i, j, 0);
			float g = get_pixel(im, i, j, 1);
			float b = get_pixel(im, i, j, 2);

			float V = three_way_max(r, g, b);
			float m = three_way_min(r, g, b);
			float C = V - m;

			float S;
			if(V != 0) {
				S = C / V;
			} else {
				S = 0;
			}

			float H;
			if(C != 0) {
				float Hprime;
				if(V == r) {
					Hprime = (g-b)/C;
				} else if(V == g) {
					Hprime = (b-r)/C + 2;
				} else if(V == b) {
					Hprime = (r-g)/C + 4;
				}

				if(Hprime < 0) {
					H = Hprime/6 + 1;
				} else {
					H = Hprime /6;
				}

				if(H < 0 || 1 <= H) {
					float intPart;
					H = modff(H, &intPart);
				}
			} else {
				H = 0;
			}

			set_pixel(im, i, j, 0, H);
			set_pixel(im, i, j, 1, S);
			set_pixel(im, i, j, 2, V);
		}
	}
}

void hsv_to_rgb(image im)
{
	for(int i = 0; i < im.w; i++) {
		for (int j = 0; j < im.h; ++j) {
			float h = get_pixel(im, i, j, 0);
			float s = get_pixel(im, i, j, 1);
			float v = get_pixel(im, i, j, 2);

			float R, G, B;
			if(v == 0) {
				R = G = B = 0;
			} else {
				float C = v * s;

				if(C == 0) {
					R = G = B = v;
				} else {
					float Hprime = 6 * h;
					assert(0 <= Hprime && Hprime <= 6);

					float max = v;
					float min = v - C;

					if (0 <= Hprime && Hprime <= 1) {
						R = max;
						G = Hprime * C + min;
						B = min;
					} else if (1 <= Hprime && Hprime <= 2) {
						R = -(C * (Hprime - 2) - min);
						G = max;
						B = min;
					} else if (2 <= Hprime && Hprime <= 3) {
						R = min;
						G = max;
						B = C * (Hprime - 2) + min;
					} else if (3 <= Hprime && Hprime <= 4) {
						R = min;
						G = -(C * (Hprime - 4) - min);
						B = max;
					} else if (5 <= Hprime && Hprime <= 6) {
						R = max;
						G = min;
						B = -((Hprime-6) * C - min);
					} else if (4 <= Hprime && Hprime <= 5) {
						R = C * (Hprime - 4) + min;
						G = min;
						B = max;
					}
				}
			}

			set_pixel(im, i, j, 0, R);
			set_pixel(im, i, j, 1, G);
			set_pixel(im, i, j, 2, B);
		}
	}
}
