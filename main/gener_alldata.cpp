//
// Created by hyj on 17-6-22.
//

#include <fstream>
#include <sys/stat.h>
#include "../src/imu.h"
#include "../src/utilities.h"
#include "../test/AutoDiff.h"
#include "../test/EigenFunction.h"

using Point = Eigen::Vector4d;
using Points = std::vector<Point, Eigen::aligned_allocator<Point> >;
using Line = std::pair<Eigen::Vector4d, Eigen::Vector4d>;
using Lines = std::vector<Line, Eigen::aligned_allocator<Line> >;



void CreatePointsLines(Points& points, Lines& lines)
{
    std::ifstream f;
#if FENCE
    f.open("../python_tool/fence.txt");
#else
    f.open("house_model/house.txt");
#endif

    while(!f.eof())
    {
        std::string s;
        std::getline(f,s);
        if(!s.empty())
        {
            std::stringstream ss;
            ss << s;
            double x,y,z;
            ss >> x;
            ss >> y;
            ss >> z;
            Eigen::Vector4d pt0( x, y, z, 1 );
            ss >> x;
            ss >> y;
            ss >> z;
            Eigen::Vector4d pt1( x, y, z, 1 );

            bool isHistoryPoint = false;
            for (int i = 0; i < points.size(); ++i) {
                Eigen::Vector4d pt = points[i];
                if(pt == pt0)
                {
                    isHistoryPoint = true;
                }
            }
            if(!isHistoryPoint)
                points.push_back(pt0);

            isHistoryPoint = false;
            for (int i = 0; i < points.size(); ++i) {
                Eigen::Vector4d pt = points[i];
                if(pt == pt1)
                {
                    isHistoryPoint = true;
                }
            }
            if(!isHistoryPoint)
                points.push_back(pt1);

            // pt0 = Twl * pt0;
            // pt1 = Twl * pt1;
            lines.emplace_back(pt0, pt1);   // lines
        }
    }
#if !FENCE
    // create more 3d points, you can comment this code
    int n = points.size();
    for (int j = 0; j < n; ++j) {
        Eigen::Vector4d p = points[j] + Eigen::Vector4d(0.5,0.5,-0.5,0);
        points.push_back(p);
    }
#endif

    // save points
    save_points("all_points.txt", points);
}

int main()
{
    // test begin
    TestEigen();
//    TestAutoDiff();
    return 0;
    // test end

    // 建立keyframe文件夹
    mkdir("keyframe", 0777);

    // 生成3d points
    Points points;
    Lines lines;
    CreatePointsLines(points, lines);

    // IMU model
    Param params;
    IMU imuGen(params);

    // create imu data
    // imu pose gyro acc
    std::vector< MotionData > imudata;
    std::vector< MotionData > imudata_noise;
    for (float t = params.t_start; t<params.t_end;) {
        MotionData data = imuGen.MotionModel(t);
        imudata.push_back(data);

        // add imu noise
        MotionData data_noise = data;
        imuGen.addIMUnoise(data_noise);
        imudata_noise.push_back(data_noise);

        t += 1.0/params.imu_frequency;
    }
    imuGen.init_velocity_ = imudata[0].imu_velocity;
    imuGen.init_twb_ = imudata.at(0).twb;
    imuGen.init_Rwb_ = imudata.at(0).Rwb;
    std::cout << "init.twb " << imuGen.init_twb_.transpose() << std::endl;
    std::cout << "init.rwb " << imuGen.init_Rwb_ << std::endl;
    save_Pose("imu_pose.txt", imudata);
    save_Pose("imu_pose_noise.txt", imudata_noise);

    imuGen.testImu("imu_pose.txt", "imu_int_pose.txt");     // test the imu data, integrate the imu data to generate the imu trajecotry
    imuGen.testImu("imu_pose_noise.txt", "imu_int_pose_noise.txt");

    // cam pose
    std::vector< MotionData > camdata;
    for (float t = params.t_start; t<params.t_end;) {

        MotionData imu = imuGen.MotionModel(t);   // imu body frame to world frame motion
        MotionData cam;

        cam.timestamp = imu.timestamp;
        cam.Rwb = imu.Rwb * params.R_bc;    // cam frame in world frame，此处用Rwb替代Rwc
        cam.twb = imu.twb + imu.Rwb * params.t_bc; //  Twc = Twb * Tbc ,  t = Rwb * tbc + twb

        camdata.push_back(cam);
        t += 1.0/params.cam_frequency;
    }
    save_Pose("cam_pose.txt",camdata);
    save_Pose_asTUM("cam_pose_tum.txt",camdata);

    // points obs in image
    int countPoint = 0;
    int countLine = 0;
    std::cout << " camdata.size() " << camdata.size() << " all " << camdata.size() * points.size()  << std::endl;
    for(int n = 0; n < camdata.size(); ++n)
    {
        MotionData data = camdata[n];
        Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
        Twc.block(0, 0, 3, 3) = data.Rwb;
        Twc.block(0, 3, 3, 1) = data.twb;

        int countValid = 0;
        // 遍历所有的特征点，看哪些特征点在视野里
        std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > points_cam;    // ３维点在当前cam视野里
        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d> > features_cam;  // 对应的２维图像坐标
        for (int i = 0; i < points.size(); ++i) {
            Eigen::Vector4d pw = points[i];          // 最后一位存着feature id
            pw[3] = 1;                               //改成齐次坐标最后一位
            Eigen::Vector4d pc1 = Twc.inverse() * pw; // T_wc.inverse() * Pw  -- > point in cam frame

            if (countPoint < 5) {
                std::cout << "\npw " << pw[0] << " "<< pw[1] << " "<< pw[2] << " " << std::endl;
                std::cout << "pc " << pc1[0] << " "<< pc1[1] << " "<< pc1[2] << " " << std::endl;
            }
            if(pc1(2) < 0) {
                countPoint++;
                continue; // z必须大于０,在摄像机坐标系前方
            }

            Eigen::Vector2d obs(pc1(0)/pc1(2), pc1(1)/pc1(2)) ;
            if( (obs(0)*params.fx + params.cx) < params.image_w && ( obs(0) * params.fx + params.cx) > 0 &&
            (obs(1)*params.fy + params.cy) > 0 && ( obs(1)* params.fy + params.cy) < params.image_h ) {
                countLine++;
                countValid++;
//                std::cout << "!!!! " << obs(0)*params.fx + params.cx << " "<< obs(0) * params.fx + params.cx << std::endl;
            }

            if (countLine < 1) {
                std::cout << "obs " << obs(0)*params.fx + params.cx << " "<< obs(1)* params.fy + params.cy << std::endl;
            }

            // if( (obs(0)*460 + 255) < params.image_h && ( obs(0) * 460 + 255) > 0 &&
                   // (obs(1)*460 + 255) > 0 && ( obs(1)* 460 + 255) < params.image_w )
//            if( (obs(0)*params.fx + params.cx) < params.image_w && ( obs(0) * params.fx + params.cx) > 0 &&
//                (obs(1)*params.fy + params.cy) > 0 && ( obs(1)* params.fy + params.cy) < params.image_h )
            {
                points_cam.push_back(points[i]);
                features_cam.push_back(obs);
            }
        }

        std::cout << " countValid " << countValid << std::endl;
        // save points
        std::stringstream filename1;
        filename1<<"keyframe/all_points_"<<n<<".txt";
        save_features(filename1.str(),points_cam,features_cam);
    }

    std::cout << " countPoint " << countPoint<< " countLine " << countLine  << std::endl;

    // lines obs in image
    for(int n = 0; n < camdata.size(); ++n)
    {
        MotionData data = camdata[n];
        Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
        Twc.block(0, 0, 3, 3) = data.Rwb;
        Twc.block(0, 3, 3, 1) = data.twb;

        // 遍历所有的特征点，看哪些特征点在视野里
        // std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > points_cam;    // ３维点在当前cam视野里
        std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > features_cam;  // 对应的２维图像坐标
        for (int i = 0; i < lines.size(); ++i) {
            Line linept = lines[i];

            Eigen::Vector4d pc1 = Twc.inverse() * linept.first; // T_wc.inverse() * Pw  -- > point in cam frame
            Eigen::Vector4d pc2 = Twc.inverse() * linept.second; // T_wc.inverse() * Pw  -- > point in cam frame

            if(pc1(2) < 0 || pc2(2) < 0)
                continue; // z必须大于０,在摄像机坐标系前方

            Eigen::Vector4d obs(pc1(0)/pc1(2), pc1(1)/pc1(2),
                                pc2(0)/pc2(2), pc2(1)/pc2(2));
            //if(obs(0) < params.image_w && obs(0) > 0 && obs(1)> 0 && obs(1) < params.image_h)
            {
                features_cam.push_back(obs);
            }
        }

        // save points
        std::stringstream filename1;
        filename1<<"keyframe/all_lines_"<<n<<".txt";
        save_lines(filename1.str(),features_cam);
    }

    return 0;
}
