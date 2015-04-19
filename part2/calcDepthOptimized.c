// CS 61C Fall 2014 Project 3

// include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

#include "calcDepthOptimized.h"
#include "calcDepthNaive.h"

void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement)
{
	int x,y;
	#pragma omp parallel for private(y)
	for (x = 0; x < imageWidth; x++)
	{
		//#pragma omp for
		for (y = 0; y < imageHeight; y++)
		{
			if ((y < featureHeight) || (y >= imageHeight - featureHeight) || (x < featureWidth) || (x >= imageWidth - featureWidth))
			{
				depth[y * imageWidth + x] = 0;
				continue;
			}

			float minimumSquaredDifference = -1;
			int minimumDy = 0;
			int minimumDx = 0;
			for (int dx = -maximumDisplacement; dx <= maximumDisplacement; dx++)
			{
				for (int dy = -maximumDisplacement; dy <= maximumDisplacement; dy++)
				{
					int temp = y + dy;
					int temp1 = x + dx;
					if (temp - featureHeight >= 0 && temp + featureHeight < imageHeight && temp1 - featureWidth >= 0 && temp1 + featureWidth < imageWidth)
					{
						

						float squaredDifference = 0;
						int boxY = 0, boxX = 0;
						int leftover = (2 * featureWidth + 1) % 8;
						__m128 sum = _mm_setzero_ps();
						__m128 sum1 = _mm_setzero_ps();
						for (boxX = -featureWidth; (boxX + 8 <= featureWidth); boxX+=8)
						{
							int leftX = x + boxX;
							int rightX = x + dx + boxX;
							for (boxY = -featureHeight; boxY <= featureHeight; boxY++)
							{	
								int index_left = (y + boxY) * imageWidth + leftX;
								int index_right = (y + dy + boxY) * imageWidth + rightX;
								__m128 left_image = _mm_loadu_ps(left + index_left);
								__m128 right_image = _mm_loadu_ps(right + index_right);
								__m128 difference = _mm_sub_ps(left_image, right_image);
								difference = _mm_mul_ps(difference, difference);

								__m128 left_image1 = _mm_loadu_ps(left + index_left+4);
								__m128 right_image1 = _mm_loadu_ps(right + index_right+4);
								__m128 difference1 = _mm_sub_ps(left_image1, right_image1);
								difference1 = _mm_mul_ps(difference1, difference1);
								sum = _mm_add_ps(_mm_add_ps(difference, difference1), sum);
							}
						}
						float sd[4] = {0,0,0,0};
						_mm_storeu_ps(sd, sum);
						squaredDifference += sd[0] + sd[1] + sd[2] + sd[3];
						boxX = featureWidth + 1 - leftover;
						sum = _mm_setzero_ps();
						if((minimumSquaredDifference != -1) || (minimumSquaredDifference < squaredDifference)){
							if (leftover == 1){
								for(boxY = -featureHeight; boxY <= featureHeight; boxY++){
									float difference = left[(y + boxY) * imageWidth + x + boxX] - right[(y + dy + boxY) * imageWidth + x + dx + boxX];
									squaredDifference += difference * difference;
								}
							} else if (leftover == 3){
								for(boxY = -featureHeight; boxY <= featureHeight; boxY++){
									__m128 leftV = _mm_loadu_ps(left + (y + boxY) * imageWidth + x + boxX);
									__m128 rightV = _mm_loadu_ps(right + (y + dy + boxY) * imageWidth + x + dx + boxX);
									__m128 difference = _mm_sub_ps(leftV, rightV);
									sum = _mm_add_ps(_mm_mul_ps(difference, difference), sum);
								}
								_mm_storeu_ps(sd, sum);
								squaredDifference += sd[0] + sd[1] + sd[2]; 
							} else if (leftover == 5){
								for(boxY = -featureHeight; boxY <= featureHeight; boxY++){
									__m128 leftV = _mm_loadu_ps(left + (y + boxY) * imageWidth + x + boxX);
									__m128 rightV = _mm_loadu_ps(right + (y + dy + boxY) * imageWidth + x + dx + boxX);
									__m128 difference = _mm_sub_ps(leftV, rightV);
									sum = _mm_add_ps(_mm_mul_ps(difference, difference), sum);
									float difference1 = left[(y + boxY) * imageWidth + x + boxX+4] - right[(y + dy + boxY) * imageWidth + x + dx + boxX+4];
									squaredDifference += difference1 * difference1;
								}
								_mm_storeu_ps(sd, sum);
								squaredDifference += sd[0] + sd[1] + sd[2] + sd[3]; 
							} else {
								for(boxY = -featureHeight; boxY <= featureHeight; boxY++){
									__m128 leftV = _mm_loadu_ps(left + (y + boxY) * imageWidth + x + boxX);
									__m128 rightV = _mm_loadu_ps(right + (y + dy + boxY) * imageWidth + x + dx + boxX);
									__m128 difference = _mm_sub_ps(leftV, rightV);
									sum = _mm_add_ps(_mm_mul_ps(difference, difference), sum);
									leftV = _mm_loadu_ps(left + (y + boxY) * imageWidth + x + boxX+4);
									rightV = _mm_loadu_ps(right + (y + dy + boxY) * imageWidth + x + dx + boxX+4);
									__m128 difference1 = _mm_sub_ps(leftV, rightV);
									sum1 = _mm_add_ps(_mm_mul_ps(difference1, difference1), sum1);
								}
								_mm_storeu_ps(sd, sum);
								squaredDifference += sd[0] + sd[1] + sd[2] + sd[3]; 
								_mm_storeu_ps(sd, sum1);
								squaredDifference += sd[0] + sd[1] + sd[2]; 
							}		
						}


						if ((minimumSquaredDifference == -1) || ((minimumSquaredDifference == squaredDifference) && (displacementNaive(dx, dy) < displacementNaive(minimumDx, minimumDy))) || (minimumSquaredDifference > squaredDifference))
						{
							minimumSquaredDifference = squaredDifference;
							minimumDx = dx;
							minimumDy = dy;
						}
					}
				}
			}

			if (minimumSquaredDifference != -1)
			{
				if (maximumDisplacement == 0)
				{
					depth[y * imageWidth + x] = 0;
				}
				else
				{
					depth[y * imageWidth + x] = displacementNaive(minimumDx, minimumDy);
				}
			}
			else
			{
				depth[y * imageWidth + x] = 0;
			}
		}
	}

}
