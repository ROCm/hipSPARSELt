#!/usr/bin/python
import itertools
import random
import sys
import getopt
import yaml
import os

class KernelArguments:
    SolutionNameMin = ""
    DataType = 4
    DestDataType = 4
    ComputeDataType = 0
    TransposeA = False
    TransposeB = False
    WorkGroup = [1, 1, 1]
    ThreadTile = [1, 1, 1]
    MacroTile = [1, 1, 0]
    StaggerU = 0
    DepthU = 32
    GlobalSplitU = 0
    StaggerStrideShift =0
    WorkGroupMapping = 0
    PackBatchDims = 0
    UseInitialStridesAB = False
    UseInitialStridesCD = False

def writefile(filename, kernel_maps):
    print(filename)
    with open(filename, 'w') as f:
        try:
            f.write("#include <map>\n")
            f.write("#include <vector>\n")
            f.write("#include <string>\n")
            f.write("#include \"hip/hip_runtime.h\"\n")
            f.write("\n")
            f.write("std::map<std::string, int> kernel_count = \n{\n")
            count_keys = len(kernel_maps.keys())
            print(count_keys)
            for key in kernel_maps.keys():
                count = len(kernel_maps[key])
                count_keys  = count_keys - 1
                if count_keys == 0:
                    f.write("    {}\"{}\", {}{}\n".format("{", key, count, "}"))
                else:
                    f.write("    {}\"{}\", {}{},\n".format("{", key, count, "}"))

            f.write("};\n")
            f.write("extern \"C\" size_t get_kernel_counts(const char* name)\n")
            f.write("{\n")
            f.write("    auto it = kernel_count.find(name);\n")
            f.write("    if(it != kernel_count.end())\n")
            f.write("        return it->second;\n")
            f.write("    else\n        return 0;\n")
            f.write("};\n")

            f.write("struct KernelParams\n")
            f.write("{\n")
            f.write("    char SolutionNameMin[256];\n")
            f.write("    int DataType;\n")
            f.write("    int DestDataType;\n")
            f.write("    int ComputeDataType;\n")
            f.write("    bool TransposeA;\n")
            f.write("    bool TransposeB;\n")
            f.write("    unsigned int WorkGroup[3];\n")
            f.write("    unsigned int ThreadTile[3];\n")
            f.write("    unsigned int MacroTile[3];\n")
            f.write("    size_t StaggerU;\n")
            f.write("    size_t DepthU;\n")
            f.write("    size_t GlobalSplitU;\n")
            f.write("    size_t StaggerStrideShift;\n")
            f.write("    int WorkGroupMapping;\n")
            f.write("    size_t PackBatchDims;\n")
            f.write("    bool UseInitialStridesAB;\n")
            f.write("    bool UseInitialStridesCD;\n")
            f.write("};\n")

            f.write("std::map<std::string, std::vector<KernelParams>> kernel_params = \n{\n")
            count_keys = len(kernel_maps.keys())
            print(count_keys)
            for key in kernel_maps.keys():
                f.write("{}\"{}\", {}".format("{", key, "{"))
                for ka in kernel_maps[key]:
                    wg = "{} {}, {}, {}{}".format("{", ka.WorkGroup[0], ka.WorkGroup[1], ka.WorkGroup[2], "}")
                    tt = "{} {}, {}, {}{}".format("{", ka.ThreadTile[0], ka.ThreadTile[1], ka.ThreadTile[2], "}")
                    mt = "{} {}, {}, {}{}".format("{", ka.MacroTile[0], ka.MacroTile[1], ka.MacroTile[2], "}")
                    values = "\"{}\", {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(
                            ka.SolutionNameMin, ka.DataType, ka.DestDataType, ka.ComputeDataType,
                            "true" if ka.TransposeA else "false", "true" if ka.TransposeB else "false",
                            wg, tt, mt,
                            ka.StaggerU, ka.DepthU, ka.GlobalSplitU, ka.StaggerStrideShift, ka.WorkGroupMapping, ka.PackBatchDims,
                            "true" if ka.UseInitialStridesA else "false", "true" if ka.UseInitialStridesCD else "false")
                    f.write("{}{}{},\n".format("{", values, "}"))
                f.write("}},\n")
            f.write("};\n")
            f.write("extern \"C\" KernelParams* get_kernel_params(const char* name)\n")
            f.write("{\n")
            f.write("    auto it = kernel_params.find(name);\n")
            f.write("    if(it != kernel_params.end())\n")
            f.write("        return it->second.data();\n")
            f.write("    else\n        return NULL;\n")
            f.write("};\n")
        except Exception as e:
            print(e)
        f.close()

def main(args):

    kernel_maps={}

    #float, half, int, bf16, int8
    dataTypes = [0, 4, 6, 7, 8]

    (opts, rem) = getopt.getopt(args, '', ['filename=', 'yaml=', 'v'])
    optDict = dict(opts)
    filename    = optDict.get('--filename', '')

    if '--yaml' in optDict:
        path = optDict['--yaml']
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        print(files)
        for file_name in files:
            file_name_u = file_name.upper()
            if ".YAML" not in file_name_u :
                continue
            with open(os.path.join(path, file_name), 'r') as f:
                try:
                    contents_a = yaml.safe_load(f)
                    print("len=", len(contents_a))
                    contents4 = contents_a[4]
                    contents5 = contents_a[5]
                    for c_index in range(0, len(contents5)):
                        ka = KernelArguments()
                        ka.SolutionNameMin = contents5[c_index].get('SolutionNameMin')
                        ka.DataType = contents4.get('DataType')
                        ka.DestDataType = contents4.get('DestDataType')
                        ka.ComputeDataType = contents4.get('ComputeDataType')
                        ka.TransposeA = contents4.get('TransposeA')
                        ka.TransposeB = contents4.get('TransposeB')
                        ka.DataType = contents4.get('DataType')
                        ka.DestDataType = contents4.get('DestDataType')
                        ka.ComputeDataType = contents4.get('ComputeDataType')
                        ka.TransposeA = contents4.get('TransposeA')
                        ka.TransposeB = contents4.get('TransposeB')
                        ka.WorkGroup = contents5[c_index].get('WorkGroup')
                        ka.ThreadTile[0] = contents5[c_index].get('ThreadTile')[0]
                        ka.ThreadTile[1] = contents5[c_index].get('ThreadTile')[1]
                        ka.MacroTile[0] = contents5[c_index].get('MacroTile0')
                        ka.MacroTile[1] = contents5[c_index].get('MacroTile1')
                        ka.StaggerU = contents5[c_index].get('StaggerU')
                        ka.DepthU = contents5[c_index].get('DepthU')
                        ka.GlobalSplitU = contents5[c_index].get('GlobalSplitU')
                        ka.StaggerStrideShift = contents5[c_index].get('_staggerStrideShift')
                        ka.WorkGroupMapping = contents5[c_index].get('WorkGroupMapping')
                        ka.PackBatchDims = contents5[c_index].get('PackBatchDims')
                        ka.UseInitialStridesA = contents4.get('UseInitialStridesAB')
                        ka.UseInitialStridesC = contents4.get('UseInitialStridesCD')
                        if '--v' in optDict:
                            print("SolutionNameMin=", ka.SolutionNameMin)
                            print("DataType=", ka.DataType)
                            print("DestDataType=", ka.DestDataType)
                            print("ComputeDataType=", ka.ComputeDataType)
                            print("TransposeA=", ka.TransposeA)
                            print("TransposeB=", ka.TransposeB)
                            print("WorkGroup=", ka.WorkGroup)
                            print("ThreadTile=", ka.ThreadTile)
                            print("MacroTile=", ka.MacroTile)
                            print("StaggerU=", ka.StaggerU)
                            print("DepthU=", ka.DepthU)
                            print("GlobalSplitU=", ka.GlobalSplitU)
                            print("StaggerStrideShift=", ka.StaggerStrideShift)
                            print("WorkGroupMapping=", ka.WorkGroupMapping)
                            print("PackBatchDims=", ka.PackBatchDims)
                            print("UseInitialStridesA=", ka.UseInitialStridesA)
                            print("UseInitialStridesC=", ka.UseInitialStridesC)
                        key="{}_{}_{}_{}_{}".format(ka.DataType, ka.DestDataType, ka.ComputeDataType, 'T' if ka.TransposeA else 'N', 'T' if ka.TransposeB else 'N')
                        if key in kernel_maps :
                            kernel_maps[key].append(ka)
                        else:
                            kas= [ka]
                            kernel_maps[key] = kas
                except Exception as e:
                    print('Failed to read file: {}'.format(filename))
                    print(e)
                    return

    print(kernel_maps["4_4_0_N_N"][0])
    writefile(filename, kernel_maps)

if __name__=="__main__":
    main(sys.argv[1:])
