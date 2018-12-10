#include "opencv_camera_display.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <VX/vx.h>
#include <VX/vx_types.h>
#include <VX/vxu.h>
#include <VX/vx_nodes.h>
#include <VX/vx_api.h>
#include <VX/vx_nodes.h>

//   ERROR_CHECK_STATUS     - check whether the status is VX_SUCCESS
#define ERROR_CHECK_STATUS( status ) { \
        vx_status status_ = (status); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
    }

//   ERROR_CHECK_OBJECT     - check whether the object creation is successful
#define ERROR_CHECK_OBJECT( obj ) { \
        vx_status status_ = vxGetStatus((vx_reference)(obj)); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
    }

////////
// log_callback() function implements a mechanism to print log messages
// from OpenVX framework onto console. The log_callback function can be
// activated by calling vxRegisterLogCallback() in STEP 02.
void VX_CALLBACK log_callback( vx_context    context,
                               vx_reference  ref,
                               vx_status     status,
                               const vx_char string[] )
{
    printf( "LOG: [ status = %d ] %s\n", status, string );
    fflush( stdout );
}

int main( int argc, char * argv[] )
{
    // Gets default video sequence when nothing is specified on command-line and
    // instantiate OpenCV GUI module for reading input RGB images and displaying
    // the image with OpenVX results.
    const char * video_sequence = argv[1];
    CGuiModule gui( video_sequence );

    // Tries to grab the first video frame from the sequence using cv::VideoCapture
    // and check if a video frame is available.
    if( !gui.Grab() )
    {
        printf( "ERROR: input has no video\n" );
        return 1;
    }

    ////////
    // Set the application configuration parameters.
    vx_uint32  width                   = gui.GetWidth();        // image width
    vx_uint32  height                  = gui.GetHeight();       // image height
    vx_size    max_keypoint_count      = 10000;                 // maximum number of keypoints to track
    vx_float32 harris_strength_thresh  = 0.0005f;               // minimum corner strength to keep a corner
    vx_float32 harris_min_distance     = 5.0f;                  // radial L2 distance for non-max suppression
    vx_float32 harris_sensitivity      = 0.04f;                 // multiplier k in det(A) - k * trace(A)^2
    vx_int32   harris_gradient_size    = 3;                     // window size for gradient computation
    vx_int32   harris_block_size       = 3;                     // block window size for Harris corner score
    vx_uint32  lk_pyramid_levels       = 6;                     // number of pyramid levels for optical flow
    vx_float32 lk_pyramid_scale        = VX_SCALE_PYRAMID_HALF; // pyramid levels scale by factor of two
    vx_enum    lk_termination          = VX_TERM_CRITERIA_BOTH; // iteration termination criteria (eps & iterations)
    vx_float32 lk_epsilon              = 0.01f;                 // convergence criterion
    vx_uint32  lk_num_iterations       = 5;                     // maximum number of iterations
    vx_bool    lk_use_initial_estimate = vx_false_e;            // don't use initial estimate
    vx_uint32  lk_window_dimension     = 6;                     // window size for evaluation
    vx_float32 trackable_kp_ratio_thr  = 0.8f;                  // threshold for the ration of tracked keypoints to all

    // Create the OpenVX context and make sure the returned context is valid.
    //
    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT( context );


    vxRegisterLogCallback( context, log_callback, vx_false_e );
    vxAddLogEntry( ( vx_reference ) context, VX_FAILURE, "Hello there!\n" );


    ////////
    // Create OpenVX image object for input RGB image.
    //
    vx_image input_rgb_image = vxCreateImage( context, width, height, VX_DF_IMAGE_RGB );
    ERROR_CHECK_OBJECT( input_rgb_image );


    // OpenVX optical flow functionality requires image pyramids for the current
    // and the previous image. It also requires keypoints that correspond
    // to the previous pyramid and will output updated keypoints into
    // another keypoint array. To be able to toggle between the current and
    // the previous buffers, you need to use OpenVX delay objects and vxAgeDelay().
    // Create OpenVX pyramid and array object exemplars and create OpenVX delay
    // objects for both to hold two of each. Note that the exemplar objects are not
    // needed once the delay objects are created.
    //
    vx_pyramid pyramidExemplar = vxCreatePyramid( context, lk_pyramid_levels,
                                                  lk_pyramid_scale, width, height, VX_DF_IMAGE_U8 );
    ERROR_CHECK_OBJECT( pyramidExemplar );
    vx_delay pyramidDelay   = vxCreateDelay( context, ( vx_reference )pyramidExemplar, 2 );
    ERROR_CHECK_OBJECT( pyramidDelay );
    ERROR_CHECK_STATUS( vxReleasePyramid( &pyramidExemplar ) );
    vx_array keypointsExemplar = vxCreateArray( context, VX_TYPE_KEYPOINT, max_keypoint_count );
    ERROR_CHECK_OBJECT( keypointsExemplar );
    vx_delay keypointsDelay = vxCreateDelay( context, ( vx_reference )keypointsExemplar, 2 );
    ERROR_CHECK_STATUS( vxReleaseArray( &keypointsExemplar ) );


    // An object from a delay slot can be accessed using vxGetReferenceFromDelay API.
    // You need to use index = 0 for the current object and index = -1 for the previous object.
    //
    vx_pyramid currentPyramid  = ( vx_pyramid ) vxGetReferenceFromDelay( pyramidDelay, 0 );
    vx_pyramid previousPyramid = ( vx_pyramid ) vxGetReferenceFromDelay( pyramidDelay, -1 );
    vx_array currentKeypoints  = ( vx_array )   vxGetReferenceFromDelay( keypointsDelay, 0 );
    vx_array previousKeypoints = ( vx_array )   vxGetReferenceFromDelay( keypointsDelay, -1 );
    ERROR_CHECK_OBJECT( currentPyramid );
    ERROR_CHECK_OBJECT( previousPyramid );
    ERROR_CHECK_OBJECT( currentKeypoints );
    ERROR_CHECK_OBJECT( previousKeypoints );


    // Harris and optical flow algorithms require their own graph objects.
    // The Harris graph needs to extract gray scale image out of input RGB,
    // compute an initial set of keypoints, and compute an initial pyramid for use
    // by the optical flow graph.
    //
    vx_graph graphHarris = vxCreateGraph( context );
    vx_graph graphTrack  = vxCreateGraph( context );
    ERROR_CHECK_OBJECT( graphHarris );
    ERROR_CHECK_OBJECT( graphTrack );


    // Harris and pyramid computation expect input to be an 8-bit image.
    // Given that input is an RGB image, it is best to extract a gray image
    // from RGB image, which requires two steps:
    //   - perform RGB to IYUV color conversion
    //   - extract Y channel from IYUV image
    vx_image harris_yuv_image       = vxCreateVirtualImage( graphHarris, width, height, VX_DF_IMAGE_IYUV );
    vx_image harris_gray_image      = vxCreateVirtualImage( graphHarris, width, height, VX_DF_IMAGE_U8 );
    vx_image opticalflow_yuv_image  = vxCreateVirtualImage( graphTrack,  width, height, VX_DF_IMAGE_IYUV );
    vx_image opticalflow_gray_image = vxCreateVirtualImage( graphTrack,  width, height, VX_DF_IMAGE_U8 );
    ERROR_CHECK_OBJECT( harris_yuv_image );
    ERROR_CHECK_OBJECT( harris_gray_image );
    ERROR_CHECK_OBJECT( opticalflow_yuv_image );
    ERROR_CHECK_OBJECT( opticalflow_gray_image );


    vx_scalar strength_thresh      = vxCreateScalar( context, VX_TYPE_FLOAT32, &harris_strength_thresh );
    vx_scalar min_distance         = vxCreateScalar( context, VX_TYPE_FLOAT32, &harris_min_distance );
    vx_scalar sensitivity          = vxCreateScalar( context, VX_TYPE_FLOAT32, &harris_sensitivity );
    vx_scalar epsilon              = vxCreateScalar( context, VX_TYPE_FLOAT32, &lk_epsilon );
    vx_scalar num_iterations       = vxCreateScalar( context, VX_TYPE_UINT32,  &lk_num_iterations );
    vx_scalar use_initial_estimate = vxCreateScalar( context, VX_TYPE_BOOL,    &lk_use_initial_estimate );
    ERROR_CHECK_OBJECT( strength_thresh );
    ERROR_CHECK_OBJECT( min_distance );
    ERROR_CHECK_OBJECT( sensitivity );
    ERROR_CHECK_OBJECT( epsilon );
    ERROR_CHECK_OBJECT( num_iterations );
    ERROR_CHECK_OBJECT( use_initial_estimate );


    vx_node nodesHarris[] =
    {
        vxColorConvertNode( graphHarris, input_rgb_image, harris_yuv_image ),
        vxChannelExtractNode( graphHarris, harris_yuv_image, VX_CHANNEL_Y, harris_gray_image ),
        vxGaussianPyramidNode( graphHarris, harris_gray_image, currentPyramid ),
        vxHarrisCornersNode( graphHarris, harris_gray_image, strength_thresh, min_distance, sensitivity, harris_gradient_size, harris_block_size, currentKeypoints, NULL )
    };
    for( vx_size i = 0; i < sizeof( nodesHarris ) / sizeof( nodesHarris[0] ); i++ )
    {
        ERROR_CHECK_OBJECT( nodesHarris[i] );
        ERROR_CHECK_STATUS( vxReleaseNode( &nodesHarris[i] ) );
    }
    ERROR_CHECK_STATUS( vxReleaseImage( &harris_yuv_image ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &harris_gray_image ) );
    ERROR_CHECK_STATUS( vxVerifyGraph( graphHarris ) );


    vx_node nodesTrack[] =
    {
        vxColorConvertNode( graphTrack, input_rgb_image, opticalflow_yuv_image ),
        vxChannelExtractNode( graphTrack, opticalflow_yuv_image, VX_CHANNEL_Y, opticalflow_gray_image ),
        vxGaussianPyramidNode( graphTrack, opticalflow_gray_image, currentPyramid ),
        vxOpticalFlowPyrLKNode( graphTrack, previousPyramid, currentPyramid,
                                            previousKeypoints, previousKeypoints, currentKeypoints,
                                            lk_termination, epsilon, num_iterations,
                                            use_initial_estimate, lk_window_dimension )
    };
    for( vx_size i = 0; i < sizeof( nodesTrack ) / sizeof( nodesTrack[0] ); i++ )
    {
        ERROR_CHECK_OBJECT( nodesTrack[i] );
        ERROR_CHECK_STATUS( vxReleaseNode( &nodesTrack[i] ) );
    }
    ERROR_CHECK_STATUS( vxReleaseImage( &opticalflow_yuv_image ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &opticalflow_gray_image ) );
    ERROR_CHECK_STATUS( vxVerifyGraph( graphTrack ) );


    ////////
    // Process the video sequence frame by frame until the end of sequence or aborted.
    for( int frame_index = 0; !gui.AbortRequested(); frame_index++ )
    {
        vx_rectangle_t cv_rgb_image_region;
        cv_rgb_image_region.start_x    = 0;
        cv_rgb_image_region.start_y    = 0;
        cv_rgb_image_region.end_x      = width;
        cv_rgb_image_region.end_y      = height;

        vx_imagepatch_addressing_t cv_rgb_image_layout;
        cv_rgb_image_layout.stride_x   = 3;
        cv_rgb_image_layout.stride_y   = gui.GetStride();

        vx_uint8 * cv_rgb_image_buffer = gui.GetBuffer();
        ERROR_CHECK_STATUS( vxCopyImagePatch( input_rgb_image, &cv_rgb_image_region, 0,
                                              &cv_rgb_image_layout, cv_rgb_image_buffer,
                                              VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST ) );


        ERROR_CHECK_STATUS( vxProcessGraph( frame_index == 0 ? graphHarris : graphTrack ) );


        vx_size num_corners = 0, num_tracking = 0;
        previousKeypoints = ( vx_array )vxGetReferenceFromDelay( keypointsDelay, -1 );
        currentKeypoints  = ( vx_array )vxGetReferenceFromDelay( keypointsDelay, 0 );
        ERROR_CHECK_OBJECT( currentKeypoints );
        ERROR_CHECK_OBJECT( previousKeypoints );
        ERROR_CHECK_STATUS( vxQueryArray( previousKeypoints, VX_ARRAY_NUMITEMS, &num_corners, sizeof( num_corners ) ) );
        if( num_corners > 0 )
        {
            vx_size kp_old_stride, kp_new_stride;
            vx_map_id kp_old_map, kp_new_map;
            vx_uint8 * kp_old_buf, * kp_new_buf;
            ERROR_CHECK_STATUS( vxMapArrayRange( previousKeypoints, 0, num_corners, &kp_old_map,
                                                 &kp_old_stride, ( void ** ) &kp_old_buf,
                                                 VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0 ) );
            ERROR_CHECK_STATUS( vxMapArrayRange( currentKeypoints, 0, num_corners, &kp_new_map,
                                                  &kp_new_stride, ( void ** ) &kp_new_buf,
                                                  VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0 ) );
            for( vx_size i = 0; i < num_corners; i++ )
            {
                vx_keypoint_t * kp_old = (vx_keypoint_t *) ( kp_old_buf + i * kp_old_stride );
                vx_keypoint_t * kp_new = (vx_keypoint_t *) ( kp_new_buf + i * kp_new_stride );
                if( kp_new->tracking_status )
                {
                    num_tracking++;
                    gui.DrawArrow( kp_old->x, kp_old->y, kp_new->x, kp_new->y );
                }
            }
            ERROR_CHECK_STATUS( vxUnmapArrayRange( previousKeypoints, kp_old_map ) );
            ERROR_CHECK_STATUS( vxUnmapArrayRange( currentKeypoints, kp_new_map ) );
        }


        ERROR_CHECK_STATUS( vxAgeDelay( pyramidDelay ) );
        ERROR_CHECK_STATUS( vxAgeDelay( keypointsDelay ) );


        char text[128];
        sprintf( text, "Keyboard ESC/Q-Quit SPACE-Pause [FRAME %d]", frame_index );
        gui.DrawText( 0, 16, text );
        sprintf( text, "Number of Corners: %d [tracking %d]", ( int )num_corners, ( int )num_tracking );
        gui.DrawText( 0, 36, text );
        gui.Show();
        if( !gui.Grab() )
        {
            // Terminate the processing loop if the end of sequence is detected.
            gui.WaitForKey();
            break;
        }
    }

    vx_perf_t perfHarris = { 0 }, perfTrack = { 0 };
    ERROR_CHECK_STATUS( vxQueryGraph( graphHarris, VX_GRAPH_PERFORMANCE, &perfHarris, sizeof( perfHarris ) ) );
    ERROR_CHECK_STATUS( vxQueryGraph( graphTrack, VX_GRAPH_PERFORMANCE, &perfTrack, sizeof( perfTrack ) ) );
    printf( "GraphName NumFrames Avg(ms) Min(ms)\n"
            "Harris    %9d %7.3f %7.3f\n"
            "Track     %9d %7.3f %7.3f\n",
            ( int )perfHarris.num, ( float )perfHarris.avg * 1e-6f, ( float )perfHarris.min * 1e-6f,
            ( int )perfTrack.num,  ( float )perfTrack.avg  * 1e-6f, ( float )perfTrack.min  * 1e-6f );


    ERROR_CHECK_STATUS( vxReleaseGraph( &graphHarris ) );
    ERROR_CHECK_STATUS( vxReleaseGraph( &graphTrack ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &input_rgb_image ) );
    ERROR_CHECK_STATUS( vxReleaseDelay( &pyramidDelay ) );
    ERROR_CHECK_STATUS( vxReleaseDelay( &keypointsDelay ) );
    ERROR_CHECK_STATUS( vxReleaseScalar( &strength_thresh ) );
    ERROR_CHECK_STATUS( vxReleaseScalar( &min_distance ) );
    ERROR_CHECK_STATUS( vxReleaseScalar( &sensitivity ) );
    ERROR_CHECK_STATUS( vxReleaseScalar( &epsilon ) );
    ERROR_CHECK_STATUS( vxReleaseScalar( &num_iterations ) );
    ERROR_CHECK_STATUS( vxReleaseScalar( &use_initial_estimate ) );
    ERROR_CHECK_STATUS( vxReleaseContext( &context ) );


    return 0;
}
