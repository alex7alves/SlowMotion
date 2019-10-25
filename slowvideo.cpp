/* Generate slow motion video.
   Author : Alex Alves.

   Compile : Make SlowVideo
   Run    : ./SlowVideo Video Stepframes
   i.e  :   ./SlowVideo original.mp4 0.2
*/

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

void AddPoints(vector<Point2f>& p, Size size);
void Delaunay( Mat& img, Subdiv2D& subdiv,vector<Point2f>& vertices);
void Points_triangulation( Subdiv2D& subdiv, vector<Point2f> p,vector<Point2f>::iterator it,vector<uchar>::iterator it_status,vector<uchar> status,Size size);
bool Inside(Rect roi, Mat m);
void MorphTriangle(Mat &img1, Mat &img2,Mat &Morph, vector<Point2i> triangle1, vector<Point2i> triangle2, vector<Point2i> triangleMorph,float a);

int main( int argc, char** argv)
{
  string videoName;
  float step;
  if( argc>2 ) {
    videoName = argv[1];
    step = atof(argv[2]);
  }else {
    cout << " Invalid arguments "<< endl;
    return -1;
  }
  int numberFrames, count=0;
  int width, height,fps;
  VideoCapture cap(videoName);
  width  = cap.get(CV_CAP_PROP_FRAME_WIDTH);
  height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
  numberFrames= cap.get(CV_CAP_PROP_FRAME_COUNT);
  fps = cap.get(CV_CAP_PROP_FPS);
  VideoWriter slow("Motion.avi", CV_FOURCC('M','J','P','G'), fps, Size(width,height));

  cout << " NUmber of frames : " << numberFrames << endl;

  if(!cap.isOpened()){
      cout << "Error opening video" << endl;
      return -1;
  }


  Mat img1,img2, gray1, gray2;

  cap >> img2;

  Mat motion(img2.rows,img2.cols,img2.type());

  Size winSize(31,31);

  // Rectangle to be used with Subdiv2D
  Size size = img2.size();
  Rect rect(0, 0, size.width, size.height);

  vector<Point2f>::iterator it;
  vector<uchar>::iterator it_status;

  TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS,20,0.03);

  const int MAX_COUNT = 500;

  while(1){

    vector<Point2f> vertices[2],v;
    vector<Point2f> points[2];
    vector<uchar> status;
    vector<float> err;
    Subdiv2D subdiv(rect);
    count++;
    if(count < numberFrames){
      cap >> img2;
    } else {
      break;
    }

    if(img1.data){
      cvtColor(img1, gray1, CV_BGR2GRAY);
      cvtColor(img2, gray2, CV_BGR2GRAY);

      goodFeaturesToTrack(gray1, points[0], MAX_COUNT,0.0001,15,Mat(),3,0, 0.01);

      calcOpticalFlowPyrLK(gray1,gray2,points[0],points[1],status, err, winSize,
      3, termcrit,0, 0.001);

      Points_triangulation(subdiv,points[0], it,it_status,status,size);

      Delaunay( img1, subdiv,vertices[0]);

      AddPoints(points[0],size);
      AddPoints(points[1],size);

      // Pick up the corresponding points
      for(size_t i=0; i<vertices[0].size(); i++){
        for(size_t j=0; j<points[0].size(); j++){
          if(vertices[0][i]==points[0][j]){
            vertices[1].push_back(points[1][j]);
          }
        }
      }


      Mat im1 = img1.clone();
      Mat im2 = img2.clone();
      im1.convertTo(im1, CV_32FC3, 1/255.0);
      im2.convertTo(im2, CV_32FC3, 1/255.0);
      for(float a=0.0;a<1.0;a=a+step) {
        vector<Point2f> v;
        for(size_t i=0; i<vertices[0].size(); i++){
          Point2i p = ((1.0-a)*vertices[0][i] + a*vertices[1][i]);
          v.push_back(p);
        }

        Mat motion = Mat::ones(img1.size(), CV_32FC3);
        motion = Scalar(1.0,1.0,1.0);

        for(size_t i=0; i<vertices[0].size(); i=i+3){
          vector<Point2i> t1,t2,tm;
          for(int j=0; j<3; j++){
            t1.push_back(vertices[0][j+i]);
            t2.push_back(vertices[1][j+i]);
            tm.push_back(v[j+i]);
          }

          // Warp all pixels inside triangle
          MorphTriangle(im1,im2,motion, t1,t2,tm,a);
        }
        motion.convertTo(motion, CV_8UC3, 255.0);

        for (int i=0; i<motion.rows; i++)
        {
          for (int j=0; j<motion.cols; j++)
          {
            Vec3b c = motion.at<Vec3b>(i,j);
            if(c[0]==255 && c[1]==255 && c[2]==255) {
              motion.at<Vec3b>(i,j)= ((1.0-a)*img1.at<Vec3b>(i,j) +(a)*img2.at<Vec3b>(i,j));
            }
          }
        }

        slow << motion;
        cout << "frame " << count << endl;

      }
    }

    swap(img2,img1);

  }
  cout << "Finished" << endl;
  cap.release();
  slow.release();
  destroyAllWindows();

  return 0;
}

void AddPoints(vector<Point2f>& p, Size size)
{
  p.push_back(Point2f(0,0));
  p.push_back(Point2f(size.width-1,0));
  p.push_back(Point2f(0,size.height-1));
  p.push_back(Point2f(size.width-1,size.height-1));
}

// Delaunay triangulation
void Delaunay( Mat& img, Subdiv2D& subdiv,
  vector<Point2f>& vertices){

  vector<Vec6f> triangleList;
  subdiv.getTriangleList(triangleList);
  vector<Point> pt(3);
  Size size = img.size();
  Rect rect(0,0, size.width, size.height);

  for( size_t i = 0; i < triangleList.size(); i++ ) {
    Vec6f t = triangleList[i];
    pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
    pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
    pt[2] = Point(cvRound(t[4]), cvRound(t[5]));

    if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2])){

        vertices.push_back(pt[0]);
        vertices.push_back(pt[1]);
        vertices.push_back(pt[2]);
    }
  }

}

void Points_triangulation( Subdiv2D& subdiv, vector<Point2f> p,vector<Point2f>::iterator it,
vector<uchar>::iterator it_status,vector<uchar> status,Size size)
{

  for( it = p.begin(), it_status = status.begin(); it != p.end(); it++, it_status++){
    if ((int) *it_status > 0){
      subdiv.insert(*it);
    }
  }

  subdiv.insert(Point2f(0,0));
  subdiv.insert(Point2f(size.width-1,0));
  subdiv.insert(Point2f(0,size.height-1));
  subdiv.insert(Point2f(size.width-1,size.height-1));

}

bool Inside(Rect roi, Mat m)
{
  if((0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols && 0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows) ){
    return true;
  }else{
    return false;
  }
}



void MorphTriangle(Mat &img1, Mat &img2,Mat &Morph, vector<Point2i> triangle1, vector<Point2i> triangle2, vector<Point2i> triangleMorph,float a)
{

  // Find bounding rectangle for each triangle
  Rect r1 = boundingRect(triangle1);
  Rect r2 = boundingRect(triangle2);
  Rect rm = boundingRect(triangleMorph);

  if(Inside(r1,img1) && Inside(r2,img1) && Inside(rm,img1) ) {

      // Delimit  triangle in region of bounding rectangle
      vector<Point2f> cropTriangle, cropTriangle2, cropTriangleMorph2;
      vector<Point> cropTriangleMorph;
      for(int i = 0; i < 3; i++)
      {
          cropTriangle.push_back( Point2f( triangle1[i].x - r1.x, triangle1[i].y -  r1.y) );
          cropTriangle2.push_back( Point2f( triangle2[i].x - r2.x, triangle2[i].y - r2.y) );

          // For use in fillConvexPoly
          cropTriangleMorph.push_back( Point((int)(triangleMorph[i].x - rm.x), (int)(triangleMorph[i].y - rm.y)) );
          // For use in getAffineTransform
          cropTriangleMorph2.push_back( Point2f((triangleMorph[i].x - rm.x), (triangleMorph[i].y - rm.y)) );
      }

      Mat cropImg1;
      img1(r1).copyTo(cropImg1);

      Mat cropImg2;
      img2(r2).copyTo(cropImg2);

      // Find the affine transform.
      Mat warpMat1 = getAffineTransform( cropTriangle,cropTriangleMorph2);
      Mat warpMat2 = getAffineTransform( cropTriangle2,cropTriangleMorph2);

      // Tem que ser do mesmo tamanho no warpAffine
      Mat cropImgMorph = Mat::zeros(rm.height, rm.width, cropImg1.type());
      Mat cropImgMorph2 = Mat::zeros(rm.height, rm.width, cropImg1.type());

      warpAffine( cropImg1,cropImgMorph, warpMat1, cropImgMorph.size(), INTER_LINEAR, BORDER_REFLECT_101);
      warpAffine( cropImg2,cropImgMorph2, warpMat2, cropImgMorph.size(), INTER_LINEAR, BORDER_REFLECT_101);

      // Get mask by filling triangle
      Mat mask = Mat::zeros(rm.height, rm.width, CV_32FC3);
      fillConvexPoly(mask, cropTriangleMorph, Scalar(1.0, 1.0, 1.0), 16, 0);

      // Somente a regiao delimitada pelo triangulo eh copiada

      multiply(cropImgMorph,mask,cropImgMorph);
      multiply(cropImgMorph2,mask,cropImgMorph2);

      fillConvexPoly(Morph(rm),cropTriangleMorph, Scalar(0, 0, 0), 16, 0);

      Morph(rm) = Morph(rm) + ( (1.0-a)*cropImgMorph + a*cropImgMorph2);

  }
}
