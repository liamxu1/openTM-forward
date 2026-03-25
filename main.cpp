// homo3d.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <string>
#include <vector>
#include "cmdline.h"

extern std::vector<float> runCustom(cfg::HomoConfig config, std::vector<float> *rho0 = nullptr);
extern void calculateThermalProperty(cfg::HomoConfig config, const std::string &filename, const std::string &outpath, float threshold);
extern void calculateThermalProperty_vdb(cfg::HomoConfig config, const std::string &filename, const std::string &outpath);
extern void voxel2vdb(const std::string &voxel_path, const std::string &out_path, int reso);
extern void vdb2voxel(const std::string &vdb_path, const std::string &out_path, int reso);

namespace homo {
	extern std::string setPathPrefix(const std::string& fprefix);
}

void printHelp()
{
	printf("Usage: [executable] [function] [options]\n\n");
	printf("Functions:\n");
	printf("  calc [input file] [output path] [dimension] [K1] [K2] [FEM error] [threshold]\n");
	printf("  calc_vdb [input file] [output path] [dimension * 3] [K1] [K2] [FEM error] [threshold]\n");
	printf("  vdb2voxel [input file] [output path] [dimension]\n");
	printf("  voxel2vdb [input file] [output path] [dimension]\n");
}

int main(int argc, char** argv)
{
	cfg::HomoConfig config;

	if (argc < 2) {
		printHelp();
		return 0;
	}
	else if (std::string(argv[1]) == "calc") {
		if (argc != 9) {
			printHelp();
			return 0;
		}
		std::string filename = argv[2];
		std::string outpath = argv[3];
		int dimension = std::stoi(argv[4]);
		double K1 = std::stod(argv[5]);
		double K2 = std::stod(argv[6]);
		double fem_error = std::stod(argv[7]);
		float threshold = std::stof(argv[8]);

		config.reso[0] = dimension;
		config.reso[1] = dimension;
		config.reso[2] = dimension;
		config.heatRatio[0] = K1;
		config.heatRatio[1] = K2;
		config.femRelThres = fem_error;

		calculateThermalProperty(config, filename, outpath, threshold);
		return 0;
	}
	else if (std::string(argv[1]) == "calc_vdb") {
		if (argc != 10) {
			printHelp();
			return 0;
		}
		std::string filename = argv[2];
		std::string outpath = argv[3];
		double K1 = std::stod(argv[7]);
		double K2 = std::stod(argv[8]);
		double fem_error = std::stod(argv[9]);

		config.reso[0] = std::stoi(argv[4]);
		config.reso[1] = std::stoi(argv[5]);
		config.reso[2] = std::stoi(argv[6]);
		config.heatRatio[0] = K1;
		config.heatRatio[1] = K2;
		config.femRelThres = fem_error;

		calculateThermalProperty_vdb(config, filename, outpath);
		return 0;
	}
	else if (std::string(argv[1]) == "vdb2voxel") {
		if (argc != 5) {
			printHelp();
			return 0;
		}
		std::string filename = argv[2];
		std::string outpath = argv[3];
		int res = std::stoi(argv[4]);

		vdb2voxel(filename, outpath, res);
		return 0;
	}
	else if (std::string(argv[1]) == "voxel2vdb") {
		if (argc != 5) {
			printHelp();
			return 0;
		}
		std::string filename = argv[2];
		std::string outpath = argv[3];
		int res = std::stoi(argv[4]);

		voxel2vdb(filename, outpath, res);
		return 0;
	}
	else {
		printHelp();
		return 0;
	}
}


