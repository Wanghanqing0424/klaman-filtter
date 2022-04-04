#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include<opencv2/video/tracking.hpp>


//HSV
    using namespace cv;
using namespace std;


Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
int g_nThresh = 100;//137
//对过滤颜色的范围的确定
int iLowH = 9;
int iHighH = 31;
int iLowS = 135;
int iHighS = 255;
int iLowV = 231;
int iHighV = 255;


int main(int argc, char** argv) {
    VideoCapture capture("/home/whq/1.mp4"); //capture the video from web cam
    Mat grayimage;
    Mat imgHsv;
    Mat frame;
    double x,vx,ax,y,vy,ay;//横坐标，速度，加速度，纵坐标，速度，加速度
    int deltaT = 1;

    Mat Q,R,P,H,Qd,Rd,Pd,Hd;//误差，误差，协方差矩阵
    Q=1e10*Mat::eye(Size(6,6),CV_64FC1);//这个多调，Q越大表示越相信测量值
    R=1e-1*Mat::eye(Size(2,2),CV_64FC1);
    P=Mat::zeros(Size(6,6),CV_64FC1);
    H=(Mat_<double>(2,6)<<1,0,0,0,0,0,
                                                                         0,0,0,1,0,0);//测量矩阵




    while (true) {

        capture >> frame;
        if (waitKey(70) != 'q') {

            //区分颜色*****************************************************************************************************
            vector<Mat> hsvSplit;//cevtor 容器

            cvtColor(frame, imgHsv, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
            cvtColor(frame, grayimage, COLOR_BGR2GRAY);//convert to gray




            split(imgHsv, hsvSplit);//split channels

            for (int i = 0; i < frame.rows; i++) {

                for (int j = 0; j < frame.cols; j++) {
                    grayimage.at<uchar>(i, j) = 0;
                    if (hsvSplit[0].at<uchar>(i, j) > iLowH && hsvSplit[0].at<uchar>(i, j) < iHighH &&
                        hsvSplit[1].at<uchar>(i, j) > iLowS && hsvSplit[2].at<uchar>(i, j) > iLowV) {
                        grayimage.at<uchar>(i, j) = 255;
                    }

                }
            }
            //开操作 (去除一些噪点)
            morphologyEx(grayimage, grayimage, MORPH_OPEN, element);

            //闭操作 (连接一些连通域)
            morphologyEx(grayimage, grayimage, MORPH_CLOSE, element);
            dilate(grayimage, grayimage, element);
            dilate(grayimage, grayimage, element);


            vector<vector<Point>> contours;//找轮廓
            findContours(grayimage, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
            vector<Rect> boundRect(contours.size());
           // Point2f Rect[4];
            Point2f center;
            char lidar_char1[10];
            char lidar_char2[10];
            for (int i = 1; i < contours.size(); i++) {
                boundRect[i] = boundingRect(contours[i]);
                // 在yuantu上绘制正外接矩形
                rectangle(frame, boundRect[i], Scalar(0, 255, 0), 2);


                center.x = (boundRect[i].x + boundRect[i].width) / 2;
                center.y = (boundRect[i].y + boundRect[i].height) / 2;
                //cout<<"x = "<<center.x<<"\t"<<"y= "<<center.y<<endl;
              //  cout << contours.size() << endl;
                sprintf(lidar_char1, "(%.1f,", center.x);
                sprintf(lidar_char2, "%.1f)", center.y);
                string lidar_str = string("xy:") + lidar_char1 + lidar_char2;
                putText(frame,                  // 图像矩阵
                        lidar_str,                  // string型文字内容
                        Point(center.x + 50, center.y + 50),           // 文字坐标，以左下角为原点
                        cv::FONT_HERSHEY_SIMPLEX,   // 字体类型
                        2,                    // 字体大小
                        Scalar(0, 0, 255), 4, 8);



             //   Point2f  position ;
             //   position.x=boundRect[i].x;
              //  position.y=boundRect[i].y;//get the positoin of meaure

                Mat Xk=(Mat_<double> (6,1)<<x,vx,ax,y,vy,ay);//initial
                Mat A=(Mat_<double> (6,6)<<
                        1,deltaT,0.5*deltaT*deltaT,0,0,0,
                        0,1,deltaT,0,0,0,
                        0,0,1,0,0,0,
                        0,0,0,1,deltaT,0.5*deltaT*deltaT,
                        0,0,0,0,1,deltaT,
                        0,0,0,0,0,1);//测量矩阵

                Mat predict_Xk=A*Xk;//1
                Mat P_temp=A*P*A.t()+Q;//2
                    Point2f  pos_pre;

                    pos_pre.x = predict_Xk.at<double>(0,0);

                pos_pre.y = predict_Xk.at<double>(3,0);

                Mat Z=(Mat_<double> (2,1)<< boundRect[i].x,boundRect[i].y);

                Mat K=P_temp*H.t()*(H*P_temp*H.t()+R).inv();//3


                predict_Xk = predict_Xk+K*(Z-H*predict_Xk);//4

                P=(Mat::eye(6,6,CV_64FC1)-K*H)*P_temp;//5

                x=predict_Xk.at<double>(0,0);
                vx=predict_Xk.at<double>(1,0);
                ax=predict_Xk.at<double>(2,0);
                y=predict_Xk.at<double>(3,0);
                vy=predict_Xk.at<double>(4,0);
                ay=predict_Xk.at<double>(5,0);



                double  T;
                T=1.2;//预测1.2秒后的位置
                A=(Mat_<double> (6,6)<<   1,T,0.5*T*T,0,0,0,
                        0,1,T,0,0,0,
                        0,0,1,0,0,0,
                        0,0,0,1,T,0.5*T*T,
                        0,0,0,0,1,T,
                        0,0,0,0,0,1);

                predict_Xk=A*predict_Xk;
                Point2f  pos;
                pos.x =predict_Xk.at<double>(0,0);
                pos.y =predict_Xk.at<double>(3,0);
               Rect pred_location;
            //   pred_location.x=pos.x;
            //   pred_location.y = pos.y;
              // pred_location.width=boundRect[i].width;
              // pred_location.height=boundRect[i].height;


              pred_location=boundRect[i];
                pred_location.x=pos.x;
                pred_location.y = pos.y;
               cout<<pred_location.x<<"\t"<<pred_location.y<<endl;
               rectangle(frame, pred_location, Scalar(255, 255, 255), 2);
               // circle(frame,pos,80,Scalar(255,255,255),-1);
            }
       //  imshow("1",grayimage);
           imshow("xiaoguo",frame);

        }
        else
            break;
    }


    return 0;


}
