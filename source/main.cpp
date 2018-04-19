#include "../include/imgFeat.h"

int main(int argc, char** argv)
{
	Mat image = imread(argv[1]);
	int blockSize = atoi(argv[2]);
	int kSize = atoi(argv[3]);
	double alpha = (double)atof(argv[4]);

	Mat cornerMap;
	if (argc == 5)
	{
		feat::detectHarrisCornersOpencv(image ,cornerMap, blockSize, kSize, alpha);
	}
	else
	{
		feat::detectHarrisLaplace(image, cornerMap);
	}
	feat::drawCornerOnImage(image, cornerMap);
	
	namedWindow("corners");
	imshow("corners", image);
	waitKey();
	return 0;
}
