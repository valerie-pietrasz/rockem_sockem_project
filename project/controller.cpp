// This example application loads a URDF world file and simulates two robots
// with physics and contact in a Dynamics3D virtual world. A graphics model of it is also shown using
// Chai3D.

#include "Sai2Model.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include "Sai2Primitives.h"

#include <iostream>
#include <string>

#include <signal.h>
bool runloop = true;
void sighandler(int sig)
{ runloop = false; }

using namespace std;
using namespace Eigen;

const string robot_file = "./resources/toro.urdf";
const string bag_file = "./resources/punching_bag.urdf";


// redis keys:
// - read:
const std::string JOINT_ANGLES_KEY = "sai2::cs225a::project::sensors::q";
const std::string JOINT_VELOCITIES_KEY = "sai2::cs225a::project::sensors::dq";
const std::string JOINT_TORQUES_SENSED_KEY; // Need to set in order to use
// - write
const std::string JOINT_TORQUES_COMMANDED_KEY = "sai2::cs225a::project::actuators::fgc";
const std::string PUNCHING_BAG_COMMANDED_KEY = "sai2::cs225a::project::actuators::bag";
const std::string OPERATIONAL_POSITION_RF = "sai2::cs225a::project::operational_position_RF";
const std::string OPERATIONAL_POSITION_LF = "sai2::cs225a::project::operational_position_LF";
const std::string OPERATIONAL_ROTATION_RF = "sai2::cs225a::project::operational_rotation_RF";
const std::string OPERATIONAL_ROTATION_LF = "sai2::cs225a::project::operational_rotation_LF";

// define states:
#define NEUTRAL 0
#define JAB_INIT 1
#define JAB_FOLLOW 2

// - model
const std::string MASSMATRIX_KEY;
const std::string CORIOLIS_KEY;
const std::string ROBOT_GRAVITY_KEY;

unsigned long long controller_counter = 0;

const bool inertia_regularization = true;

// Function prototypes //
VectorXd orthodox_posture(VectorXd q_desired);
VectorXd jab_posture(VectorXd q_desired);
Vector3d perturb_bag(Vector3d q_);

//--------------------------------------- Main ---------------------------------------//
//--------------------------------------- Main ---------------------------------------//
//--------------------------------------- Main ---------------------------------------//

int main() {

	// start redis client
	auto redis_client = RedisClient();
	redis_client.connect();

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load robots
	auto robot = new Sai2Model::Sai2Model(robot_file, false);
	robot->_q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
	VectorXd initial_q = robot->_q;
	robot->updateModel();

	auto bag = new Sai2Model::Sai2Model(bag_file, false);
	Vector3d bag_torques = Vector3d(0, 0, 0);
	bag->updateModel();
	// redis_client.setEigenMatrixJSON(PUNCHING_BAG_COMMANDED_KEY, bag_torques); // Don't know why I thought I needed this line? -- Val

	// prepare controller
	int dof = robot->dof();
	VectorXd command_torques = VectorXd::Zero(dof);
	MatrixXd N_prec = MatrixXd::Identity(dof, dof);

	// Edit kp and kv values
	double kp_foot = 200;
	double kv_foot = 20;
	double kp_hand = 200;
	double kv_hand = 20;
	double kp_head = 25;
	double kv_head = 10;

	// record initial position of torso
	string hip_control_link = "hip_base";
	Vector3d x_pos_hip_init;
	Vector3d hip_control_point = Vector3d(0,0,0);
	robot->positionInWorld(x_pos_hip_init, hip_control_link, hip_control_point);

	// pose task for right foot
	string control_link = "RL_foot";
	Vector3d control_point = Vector3d(0,0,0);
	auto posori_task_footR = new Sai2Primitives::PosOriTask(robot, control_link, control_point);
	posori_task_footR->setDynamicDecouplingFull();

	// #ifdef USING_OTG
	// 	posori_task_footR->_use_interpolation_flag = true;
	// #else
	// 	posori_task_footR->_use_velocity_saturation_flag = true;
	// #endif

	VectorXd posori_task_torques_footR = VectorXd::Zero(dof);
	posori_task_footR->_kp_pos = kp_foot;
	posori_task_footR->_kv_pos = kv_foot;
	posori_task_footR->_kp_ori = kp_foot;
	posori_task_footR->_kv_ori = kv_foot;

	// set desired position and orientation to the initial configuration
	Vector3d x_pos;
	robot->positionInWorld(x_pos, control_link, control_point);
	Matrix3d x_ori;
	robot->rotationInWorld(x_ori, control_link);
	// posori_task_footR->_desired_position = x_pos;
	posori_task_footR->_desired_position = Vector3d(-0.296713, -0.268187, -1.065379);
	posori_task_footR->_desired_orientation = x_ori;

	// pose task for left foot
	control_link = "LL_foot";
	control_point = Vector3d(0,0,0);
	auto posori_task_footL = new Sai2Primitives::PosOriTask(robot, control_link, control_point);
	posori_task_footL->setDynamicDecouplingFull();

	// #ifdef USING_OTG
	// 	posori_task_footL->_use_interpolation_flag = true;
	// #else
	// 	posori_task_footL->_use_velocity_saturation_flag = true;
	// #endif

	VectorXd posori_task_torques_footL = VectorXd::Zero(dof);
	posori_task_footL->_kp_pos = kp_foot;
	posori_task_footL->_kv_pos = kv_foot;
	posori_task_footL->_kp_ori = kp_foot;
	posori_task_footL->_kv_ori = kv_foot;

	// set desired position and orientation to the initial configuration
	robot->positionInWorld(x_pos, control_link, control_point);
	robot->rotationInWorld(x_ori, control_link);
	// posori_task_footL->_desired_position = x_pos;
	posori_task_footL->_desired_position = Vector3d(0.282772, 0.269314, -1.065379);
	posori_task_footL->_desired_orientation = x_ori;

	// pose task for right hand
	control_link = "ra_link6";
	control_point = Vector3d(0,0,0);
	auto posori_task_handR = new Sai2Primitives::PosOriTask(robot, control_link, control_point);
	posori_task_handR->setDynamicDecouplingFull();

	#ifdef USING_OTG
		posori_task_handR->_use_interpolation_flag = true;
	#else
		posori_task_handR->_use_velocity_saturation_flag = true;
	#endif

	VectorXd posori_task_torques_handR = VectorXd::Zero(dof);
	posori_task_handR->_kp_pos = kp_hand;
	posori_task_handR->_kv_pos = kv_hand;
	posori_task_handR->_kp_ori = kp_hand;
	posori_task_handR->_kv_ori = kv_hand;

	// set two goal positions/orientations

	robot->positionInWorld(x_pos, control_link, control_point);
	robot->rotationInWorld(x_ori, control_link);
	// posori_task_handR->_desired_position = x_pos + Vector3d(0.5, -0.2, 0.8);
	posori_task_handR->_desired_position = x_pos_hip_init + Vector3d(0.1, -0.1, 0.1);
	posori_task_handR->_desired_orientation = AngleAxisd(M_PI/4, Vector3d::UnitZ()).toRotationMatrix() * x_ori;
	//posori_task_handR->_desired_orientation = AngleAxisd(M_PI/2, Vector3d::UnitX()).toRotationMatrix() * \
	 											AngleAxisd(-M_PI/2, Vector3d::UnitY()).toRotationMatrix() * x_ori;

	// pose task for left hand
	control_link = "la_link6";
	control_point = Vector3d(0,0,0);
	auto posori_task_handL = new Sai2Primitives::PosOriTask(robot, control_link, control_point);
	posori_task_handL->setDynamicDecouplingFull();

	#ifdef USING_OTG
		posori_task_handL->_use_interpolation_flag = true;
	#else
		posori_task_handL->_use_velocity_saturation_flag = true;
	#endif

	VectorXd posori_task_torques_handL = VectorXd::Zero(dof);
	posori_task_handL->_kp_pos = kp_hand;
	posori_task_handL->_kv_pos = kv_hand;
	posori_task_handL->_kp_ori = kp_hand;
	posori_task_handL->_kv_ori = kv_hand;

	// set two goal positions/orientations
	robot->positionInWorld(x_pos, control_link, control_point);
	robot->rotationInWorld(x_ori, control_link);
	// posori_task_handL->_desired_position = x_pos + Vector3d(0.5, 0.2, 0.8);

	posori_task_handL->_desired_position = x_pos_hip_init + Vector3d(1, 0.5, 0.5);
	posori_task_handL->_desired_orientation = AngleAxisd(-M_PI/4, Vector3d::UnitZ()).toRotationMatrix() * x_ori;
	// posori_task_handR->_desired_orientation = AngleAxisd(M_PI/2, Vector3d::UnitX()).toRotationMatrix() * \
	// 											AngleAxisd(-M_PI/2, Vector3d::UnitY()).toRotationMatrix() * x_ori;


	// pose task for head
	control_link = "neck_link2";
	control_point = Vector3d(0,0,0);
	auto posori_task_head = new Sai2Primitives::PosOriTask(robot, control_link, control_point);
	posori_task_head->setDynamicDecouplingFull();

	#ifdef USING_OTG
		posori_task_head->_use_interpolation_flag = true;
	#else
		posori_task_head->_use_velocity_saturation_flag = true;
	#endif

	VectorXd posori_task_torques_head = VectorXd::Zero(dof);
	posori_task_head->_kp_pos = kp_head;
	posori_task_head->_kv_pos = kv_head;
	posori_task_head->_kp_ori = kp_head;
	posori_task_head->_kv_ori = kv_head;

	// set two goal positions/orientations
	robot->positionInWorld(x_pos, control_link, control_point);
	robot->rotationInWorld(x_ori, control_link);
	posori_task_head->_desired_position = x_pos;
	posori_task_head->_desired_orientation = x_ori;
	// posori_task_handR->_desired_orientation = AngleAxisd(M_PI/2, Vector3d::UnitX()).toRotationMatrix() * \
	// 											AngleAxisd(-M_PI/2, Vector3d::UnitY()).toRotationMatrix() * x_ori;

	// joint task
	auto joint_task = new Sai2Primitives::JointTask(robot);
	joint_task->setDynamicDecouplingFull();

	#ifdef USING_OTG
		joint_task->_use_interpolation_flag = true;
	#else
		joint_task->_use_velocity_saturation_flag = true;
	#endif

	VectorXd joint_task_torques = VectorXd::Zero(dof);
	joint_task->_kp = 100.0;
	joint_task->_kv = 20.0;

	// Record initial joint posture

	VectorXd q_init_desired = robot->_q;
	VectorXd q_desired = q_init_desired;

	// Set state to initial state
	//Initial state//
	int state = NEUTRAL;

	// gravity vector
	VectorXd g(dof);

	// create a timer
	LoopTimer timer;
	timer.initializeTimer();
	timer.setLoopFrequency(200);
	double start_time = timer.elapsedTime(); //secs
	bool fTimerDidSleep = true;

	// Initialize useful vectors
	Vector3d x_pos_rf;
	Vector3d x_pos_lf;
	Vector3d x_pos_rh;
	Vector3d x_pos_lh;
	Vector3d x_pos_bag;

	while (runloop) {
		// wait for next scheduled loop
		timer.waitForNextLoop();
		double time = timer.elapsedTime() - start_time;

		// read robot state from redis
		robot->_q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
		robot->_dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_KEY);

		// update model
		robot->updateModel();

		switch(state){
			case NEUTRAL:
				// Define Orthodox posture
				q_desired = q_init_desired;
				q_desired = orthodox_posture(q_desired);

				// Set joint task posture to orthodox
				joint_task->_desired_position = q_desired;

				// calculate torques to fix the feet
				N_prec.setIdentity();
				posori_task_footR->updateTaskModel(N_prec);
				posori_task_footR->computeTorques(posori_task_torques_footR);

				N_prec = posori_task_footR->_N;
				posori_task_footL->updateTaskModel(N_prec);
				posori_task_footL->computeTorques(posori_task_torques_footL);

				N_prec = posori_task_footL->_N;
				joint_task->updateTaskModel(N_prec);
				joint_task->computeTorques(joint_task_torques);

				// calculate gravity torques (if needed)
				robot->gravityVector(g);

				// calculate command torques
				command_torques = posori_task_torques_footR + posori_task_torques_footL + joint_task_torques;

				cout << (robot->_q - q_desired).squaredNorm() << endl;
				if ((robot->_q - q_desired).squaredNorm() < 0.04){
					state = JAB_INIT;
				}
				break;

			case JAB_INIT:

				bag->positionInWorld(x_pos_bag, "bag", control_point);
				x_pos_bag[2] -= 0.75;
				robot->positionInWorld(x_pos_rh, "ra_link6", control_point);

				//update posori task
				posori_task_handR->_desired_position = x_pos_bag;

				// Define Jab posture
				q_desired = q_init_desired;
				q_desired = jab_posture(q_desired);

				// Set joint task posture to jab
				joint_task->_desired_position = q_desired;

				// calculate torques to fix the feet
				N_prec.setIdentity();
				posori_task_footR->updateTaskModel(N_prec);
				posori_task_footR->computeTorques(posori_task_torques_footR);

				N_prec = posori_task_footR->_N;
				posori_task_footL->updateTaskModel(N_prec);
				posori_task_footL->computeTorques(posori_task_torques_footL);

				N_prec = posori_task_footL->_N;
				posori_task_handR->updateTaskModel(N_prec);
				posori_task_handR->computeTorques(posori_task_torques_handR);

				// calculate torques to move left hand
				N_prec = posori_task_handR->_N;

				joint_task->updateTaskModel(N_prec);
				joint_task->computeTorques(joint_task_torques);

				// calculate gravity torques (if needed)
				robot->gravityVector(g);

				// calculate command torques
				command_torques = posori_task_torques_footR + posori_task_torques_footL + posori_task_torques_handR + joint_task_torques;
				cout << (x_pos_bag - x_pos_rh).squaredNorm() << endl;

				if ((x_pos_bag - x_pos_rh).squaredNorm() < 0.05){
					state = NEUTRAL;
				}
				break;


		}

		// posori_task_handR->updateTaskModel(N_prec);
		// posori_task_handR->computeTorques(posori_task_torques_handR);

		// // calculate torques to move left hand
		// N_prec = posori_task_handR->_N;
		// posori_task_handL->updateTaskModel(N_prec);
		// posori_task_handL->computeTorques(posori_task_torques_handL);

		// // calculate torques to move head
		// N_prec = posori_task_handL->_N;
		// posori_task_head->updateTaskModel(N_prec);
		// posori_task_head->computeTorques(posori_task_torques_head);

		// // calculate torques to maintain joint posture
		// N_prec = posori_task_head->_N;
		//joint_task->updateTaskModel(N_prec);
		//joint_task->computeTorques(joint_task_torques);

		// calculate gravity torques (if needed)
		//robot->gravityVector(g);

		// calculate torques
		// command_torques = posori_task_torques_footR + posori_task_torques_footL + \
		// 					posori_task_torques_handR + posori_task_torques_handL + \
		// 					posori_task_torques_head + joint_task_torques;  // gravity compensation handled in sim
		// command_torques = posori_task_torques_footR + posori_task_torques_footL + joint_task_torques;
		// command_torques = joint_task_torques;

		// PUNCHING BAG //
		// if needed, read bag state from redis, like so:
		// robot->_q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);

		bag->updateModel();
		// if(controller_counter == 10)
		// 	bag_torques = perturb_bag(bag_torques);

		// send to redis
		redis_client.setEigenMatrixJSON(JOINT_TORQUES_COMMANDED_KEY, command_torques);
		redis_client.setEigenMatrixJSON(PUNCHING_BAG_COMMANDED_KEY, bag_torques);

		//test tracking foot position with only joint task torques to choose posori targets
		// Vector3d x_pos_rf;
		// robot->positionInWorld(x_pos_rf, "RL_foot", Vector3d(0,0,0));
		// Matrix3d x_ori_rf;
		// robot->rotationInWorld(x_ori_rf, "RL_foot");

		// Vector3d x_pos_lf;
		// robot->positionInWorld(x_pos_lf, "LL_foot", Vector3d(0,0,0));
		// Matrix3d x_ori_lf;
		// robot->rotationInWorld(x_ori_lf, "LL_foot");

		// redis_client.setEigenMatrixJSON(OPERATIONAL_POSITION_RF, x_pos_rf.transpose());
		// redis_client.setEigenMatrixJSON(OPERATIONAL_POSITION_LF, x_pos_lf.transpose());
		// redis_client.setEigenMatrixJSON(OPERATIONAL_ROTATION_RF, x_ori_rf.transpose());
		// redis_client.setEigenMatrixJSON(OPERATIONAL_ROTATION_LF, x_ori_lf.transpose());

		// std::cout << "Z actuation position : " << q_desired[2] << "\n";

		//increment
		controller_counter++;
	}

	double end_time = timer.elapsedTime();
    std::cout << "\n";
    std::cout << "Controller Loop run time  : " << end_time << " seconds\n";
    std::cout << "Controller Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Controller Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";

	return 0;
}

//------------------------------- Functions -------------------------------//
//------------------------------- Functions -------------------------------//
//------------------------------- Functions -------------------------------//

VectorXd orthodox_posture(VectorXd q_desired) {

	// Overactuation
	q_desired[2] = -0.135069;
	q_desired[5] = 0;

	// Right Leg
	q_desired[6] = -M_PI/16;
	q_desired[7] = 0;
	q_desired[8] = 0;
	q_desired[9] = M_PI/4;
	q_desired[10] = M_PI/16;
	q_desired[11] = -M_PI/4;

	// Left Leg
	q_desired[12] = M_PI/16;
	q_desired[13] = -M_PI/4;
	q_desired[14] = 0;
	q_desired[15] = M_PI/4;
	q_desired[16] = -M_PI/16;
	q_desired[17] = 0;

	// Trunk
	q_desired[18] = -M_PI/6;

	// Right Arm
	q_desired[19] = M_PI/6;
	q_desired[20] = M_PI/6;
	q_desired[21] = 0;
	q_desired[22] = 3*M_PI/4;;
	q_desired[23] = 0;
	q_desired[24] = 0;

	// Left Arm
	q_desired[25] = M_PI/6;
	q_desired[26] = M_PI/6;
	q_desired[27] = 0;
	q_desired[28] = 3*M_PI/4;;
	q_desired[29] = 0;
	q_desired[30] = 0;

	// Head
	q_desired[31] = M_PI/6;

	return q_desired;
}

VectorXd jab_posture(VectorXd q_desired) {

	// Overactuation
	q_desired[2] = -0.135069;
	q_desired[5] = 0;

	// Right Leg
	q_desired[6] = -M_PI/16;
	q_desired[7] = 0;
	q_desired[8] = 0;
	q_desired[9] = M_PI/4;
	q_desired[10] = M_PI/16;
	q_desired[11] = -M_PI/4;

	// Left Leg
	q_desired[12] = M_PI/16;
	q_desired[13] = -M_PI/4;
	q_desired[14] = 0;
	q_desired[15] = M_PI/4;
	q_desired[16] = -M_PI/16;
	q_desired[17] = 0;

	// Trunk
	q_desired[18] = 0;

	// Right Arm
	q_desired[19] = M_PI/2;
	q_desired[20] = 0;
	q_desired[21] = 0;
	q_desired[22] = 0;
	q_desired[23] = 0;
	q_desired[24] = 0;

	// Left Arm
	q_desired[25] = M_PI/6;
	q_desired[26] = M_PI/6;
	q_desired[27] = 0;
	q_desired[28] = 3*M_PI/4;;
	q_desired[29] = 0;
	q_desired[30] = 0;

	// Head
	q_desired[31] = M_PI/6;

	return q_desired;
}
