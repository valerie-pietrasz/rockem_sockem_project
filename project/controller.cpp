#include "Sai2Model.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include "Sai2Primitives.h"

#include <iostream>
#include <string>
#include <random>

#include <signal.h>
// flags for simulation and controller states
bool runloop = false;
void sighandler(int){runloop = false;}
bool fSimulationLoopDone = false;
bool fControllerLoopDone = false;

using namespace std;
using namespace Eigen;

const string robot_file = "./resources/toro.urdf";
const string bag_file = "./resources/punching_bag.urdf";

// redis keys:
// - read:
const std::string JOINT_ANGLES_KEY = "sai2::cs225a::project::sensors::q";
const std::string JOINT_VELOCITIES_KEY = "sai2::cs225a::project::sensors::dq";
const std::string PUNCHING_BAG_ANGLES_KEY = "sai2::cs225a::project::sensors::bag";
// const std::string JOINT_TORQUES_SENSED_KEY; // Need to set in order to use
// - write
const std::string JOINT_TORQUES_COMMANDED_KEY = "sai2::cs225a::project::actuators::fgc";
const std::string OPERATIONAL_POSITION_RF = "sai2::cs225a::project::operational_position_RF";
const std::string OPERATIONAL_POSITION_LF = "sai2::cs225a::project::operational_position_LF";
const std::string OPERATIONAL_ROTATION_RF = "sai2::cs225a::project::operational_rotation_RF";
const std::string OPERATIONAL_ROTATION_LF = "sai2::cs225a::project::operational_rotation_LF";
// - read + write:
const std::string SIMULATION_LOOP_DONE_KEY = "cs225a::simulation::done";
const std::string CONTROLLER_LOOP_DONE_KEY = "cs225a::controller::done";

// define states:
#define NEUTRAL_INIT 0
#define NEUTRAL 1
#define DODGE_INIT 2
#define DODGE 3
#define CROSS_INIT 4
#define CROSS 5
#define JAB_INIT 6
#define JAB 7

// - model
const std::string MASSMATRIX_KEY;
const std::string CORIOLIS_KEY;
const std::string ROBOT_GRAVITY_KEY;

const bool inertia_regularization = true;

const int nsUpdateFrequency = 50; //How often the null spaces are recalculated

// Function prototypes //
void orthodox_posture(VectorXd& q_desired);
void dodge_posture(VectorXd& q_desired);
void cross_posture(VectorXd& q_desired);
void jab_posture(VectorXd& q_desired);

// functions for converting bool <--> string
inline const char * const bool_to_string(bool b);
bool string_to_bool(const std::string& x);

//--------------------------------------- Main ---------------------------------------//
//--------------------------------------- Main ---------------------------------------//
//--------------------------------------- Main ---------------------------------------//

int main() {

	// Make sure redis-server is running at localhost with default port 6379
	// start redis client
	RedisClient redis_client = RedisClient();
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

	Matrix3d R_world_bag;
	R_world_bag = AngleAxisd(M_PI/2, Vector3d::UnitZ())
								* AngleAxisd(0.0, Vector3d::UnitY())
								* AngleAxisd(M_PI/2, Vector3d::UnitX());
	Affine3d T_world_bag = Affine3d::Identity();
	T_world_bag.translation() = Vector3d(0.8, 0, 0.82);
	T_world_bag.linear() = R_world_bag;

	auto bag = new Sai2Model::Sai2Model(bag_file, false, T_world_bag);
	bag->updateModel();
	bag->_q = Vector3d(0,0,-M_PI/12);
	bag->updateKinematics();

	//------------- Controller setup: Using full PosOriTask for head, legs, and PosiTask only for arms -------------//

	// prepare controller
	int dof = robot->dof();
	VectorXd command_torques = VectorXd::Zero(dof);
	// Vector3d bag_torques = Vector3d::Zero();
	MatrixXd N_prec = MatrixXd::Identity(dof, dof);
	MatrixXd N_prec_feet = MatrixXd::Identity(dof, dof);

	// Edit kp and kv values
	double kp_foot = 600;  // 600
	double kv_foot = 200;  // 200
	double kp_hand = 400;  // 200
	double kv_hand = 100;  // 100
	double kp_head = 25;   // 25
	double kv_head = 10;   // 10
	double kp_joint = 300; // 300
	double kv_joint = 100; // 100

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

	#ifdef USING_OTG
		posori_task_footR->_use_interpolation_flag = false;
		posori_task_footR->_use_velocity_saturation_flag = false;
	#endif

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

	#ifdef USING_OTG
		posori_task_footL->_use_interpolation_flag = false;
		posori_task_footL->_use_velocity_saturation_flag = false;
	#endif

	VectorXd posori_task_torques_footL = VectorXd::Zero(dof);
	posori_task_footL->_kp_pos = kp_foot;
	posori_task_footL->_kv_pos = kv_foot;
	posori_task_footL->_kp_ori = kp_foot;
	posori_task_footL->_kv_ori = kv_foot;

	// set desired position and orientation to the initial configuration
	robot->positionInWorld(x_pos, control_link, control_point);
	robot->rotationInWorld(x_ori, control_link);
	posori_task_footL->_desired_position = Vector3d(0.282772, 0.269314, -1.065379);
	posori_task_footL->_desired_orientation = x_ori;

	// pose task for right hand
	control_link = "ra_link6";
	control_point = Vector3d(0,0,0);
	// auto posori_task_handR = new Sai2Primitives::PosOriTask(robot, control_link, control_point);
	// posori_task_handR->setDynamicDecouplingFull();
	auto posori_task_handR = new Sai2Primitives::PositionTask(robot, control_link, control_point);

	#ifdef USING_OTG
		posori_task_handR->_use_interpolation_flag = false;
		posori_task_handR->_use_velocity_saturation_flag = false;
		posori_task_handR->_otg->setMaxAcceleration(3);
		posori_task_handR->_otg->setMaxVelocity(10);
	#endif

	VectorXd posori_task_torques_handR = VectorXd::Zero(dof);
	posori_task_handR->_kp = kp_hand;
	posori_task_handR->_kv = kv_hand;
	// posori_task_handR->_kp_ori = kp_hand;
	// posori_task_handR->_kv_ori = kv_hand;

	// set two goal positions/orientations

	robot->positionInWorld(x_pos, control_link, control_point);
	robot->rotationInWorld(x_ori, control_link);
	// posori_task_handR->_desired_position = x_pos + Vector3d(0.5, -0.2, 0.8);
	posori_task_handR->_desired_position = x_pos_hip_init + Vector3d(0.1, -0.1, 0.1);

	// pose task for left hand
	control_link = "la_link6";
	control_point = Vector3d(0,0,0);
  // auto posori_task_handL = new Sai2Primitives::PosOriTask(robot, control_link, control_point);
	// posori_task_handL->setDynamicDecouplingFull();
	auto posori_task_handL = new Sai2Primitives::PositionTask(robot, control_link, control_point);

	#ifdef USING_OTG
		posori_task_handL->_use_interpolation_flag = false;
		posori_task_handL->_use_velocity_saturation_flag = false;
		posori_task_handL->_otg->setMaxAcceleration(4);
		posori_task_handL->_otg->setMaxVelocity(6);
	#endif

	VectorXd posori_task_torques_handL = VectorXd::Zero(dof); // check out position task primitive
	posori_task_handL->_kp = kp_hand + 100; // Add 100 to amp up jab compared to cross (since it starts closer to the bag);
	posori_task_handL->_kv = kv_hand;
	// posori_task_handL->_kp_ori = kp_hand;
	// posori_task_handL->_kv_ori = kv_hand;

	// set two goal positions/orientations
	robot->positionInWorld(x_pos, control_link, control_point);
	robot->rotationInWorld(x_ori, control_link);
	// posori_task_handL->_desired_position = x_pos + Vector3d(0.5, 0.2, 0.8);

	posori_task_handL->_desired_position = x_pos_hip_init + Vector3d(1, 0.5, 0.5);

	// pose task for head
	control_link = "neck_link2";
	control_point = Vector3d(0,0,0);
	auto posori_task_head = new Sai2Primitives::PosOriTask(robot, control_link, control_point);
	posori_task_head->setDynamicDecouplingFull();

	#ifdef USING_OTG
		posori_task_head->_use_interpolation_flag = true;
	//#else
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

	// joint task
	auto joint_task = new Sai2Primitives::JointTask(robot);
	joint_task->setDynamicDecouplingFull();

	#ifdef USING_OTG
		joint_task->_use_interpolation_flag = false;
		joint_task->_use_velocity_saturation_flag = true;
		// joint_task->_otg->setMaxAcceleration(3);
		// joint_task->_otg->setMaxVelocity(6);
	#endif

	VectorXd joint_task_torques = VectorXd::Zero(dof);
	joint_task->_kp = kp_joint;
	joint_task->_kv = kv_joint;

	// Record initial joint posture
	VectorXd q_init_desired = robot->_q;
	VectorXd q_desired = q_init_desired;

	// Initial state //
	int state = DODGE_INIT;
	int state_time = -600; // Hack to make the first dodge longer than the rest
	// gravity vector
	VectorXd g(dof);

	// Initialize useful vectors
	Vector3d bag_cm = Vector3d(0, -0.8, 0);
	Vector3d bag_end = Vector3d(0, -1.4, -0.2);
	Vector3d x_pos_rf;
	Vector3d x_pos_lf;
	Vector3d x_pos_rh;
	Vector3d x_pos_lh;
	Vector3d x_pos_bag_cm;
	Vector3d x_pos_bag_end;
	Vector3d current_desired_pos;
	Vector3d noise;
	int randomPunch;

	// white noise generator
	double noise_magnitude = .05; // change
	const double mean = 0.0;
  const double stddev = noise_magnitude;  // tune based on your system
  std::default_random_engine generator;
  std::normal_distribution<double> dist(mean, stddev);

	// create a loop timer
	double control_freq = 1000;
	LoopTimer timer;
	timer.setLoopFrequency(control_freq);   // 1 KHz
	timer.initializeTimer(1000000); // 1 ms pause before starting loop
	bool fTimerDidSleep = true;
	double start_time = timer.elapsedTime(); // secs

	unsigned long long counter = 0;
	int state_timeout = 600;

	runloop = true;
	while (runloop) {

		// read simulation state
    fSimulationLoopDone = string_to_bool(redis_client.get(SIMULATION_LOOP_DONE_KEY));

		// run controller loop when simulation loop is done
		if (fSimulationLoopDone) {

			// read robot state from redis
			robot->_q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
			robot->_dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_KEY);
			bag->_q = redis_client.getEigenMatrixJSON(PUNCHING_BAG_ANGLES_KEY);

			// update model
			robot->updateModel();
			bag->updateModel();
			// calculate gravity torques (if needed)
			// robot->gravityVector(g);

			// sensing world
			bag->positionInWorld(x_pos_bag_cm, "bag", bag_cm);
			bag->positionInWorld(x_pos_bag_end, "bag", bag_end);
			noise << dist(generator), dist(generator), dist(generator);
			x_pos_bag_cm += noise/2;
			x_pos_bag_end += noise;

			robot->positionInWorld(x_pos_rh, "ra_link6");
			robot->positionInWorld(x_pos_lh, "la_link6");

			// set N_prec and calculate torques to fix feet
			if(counter % nsUpdateFrequency == 0){
				N_prec.setIdentity();
				posori_task_footR->updateTaskModel(N_prec);
				N_prec = posori_task_footR->_N;
				posori_task_footL->updateTaskModel(N_prec);
				N_prec_feet = posori_task_footL->_N;
			}

			posori_task_footR->computeTorques(posori_task_torques_footR);
			posori_task_footL->computeTorques(posori_task_torques_footL);
			/*
			Important: Create a selection matrix that we left-multiply N_prec by in our posori_task_handL/R
			to select the joints to be used in the task. The rest should remain in the nullspace of N_prec.
			*/
			MatrixXd taskProjection = MatrixXd::Zero(dof, dof);
			taskProjection(18, 18) = 1; // trunk
			taskProjection(28, 28) = 1; // left elbow

			/*
			Important: Remove elbow from N_prec in our joint task so it can move freely when we punch.
			Can also allow, for example, shoulder to move freely.
			*/
			MatrixXd jointTaskProjection = MatrixXd::Identity(dof, dof);
			jointTaskProjection(28, 28) = 0; // left elbow

			// cout << (x_pos_bag_end - x_pos_hip_init).squaredNorm() << endl;

			switch(state){

				case NEUTRAL_INIT:
					// Define Orthodox posture //
					q_desired = q_init_desired;
					orthodox_posture(q_desired);

					// Generate command torques
					joint_task->_desired_position = q_desired;
					joint_task->updateTaskModel(N_prec_feet);
					joint_task->computeTorques(joint_task_torques);

					command_torques = posori_task_torques_footR + posori_task_torques_footL + joint_task_torques;

					state = NEUTRAL;
					cout << "Neutral" << endl;
					break;

				case NEUTRAL:
				//timer
				state_time++;

				if(state_time > state_timeout) { // When recovering from a punch, we amp up the gains for a bit
					joint_task->_kp = kp_joint;
					joint_task->_kv = kv_joint;
				}

					// update the models
					if(counter % nsUpdateFrequency == 0){
						joint_task->updateTaskModel(N_prec_feet);
					}

					joint_task->computeTorques(joint_task_torques);
					command_torques = posori_task_torques_footR + posori_task_torques_footL + joint_task_torques;

					// cout << (x_pos_bag_end - x_pos_hip_init).squaredNorm() << endl;
					// 0.6964 is our steady state
					if ((x_pos_bag_end - x_pos_hip_init).squaredNorm() < 0.6964 - 0.29){
						state = DODGE_INIT;
						// start timer
						state_time = 0; // This needs to be set here or the beginning of the sim will break
						// cout << "Dodge Init" << endl;
					}

					if ((robot->_q - q_desired).squaredNorm() < 0.04){
						current_desired_pos = x_pos_bag_cm;
						randomPunch = rand() % 2;
						if (randomPunch == 0){
							state = CROSS_INIT;
							// cout << "Cross Init" << endl;
						}
						else {
							state = JAB_INIT;
							// cout << "Jab Init" << endl;
							// If we want to also control the orientation of the punch (not currently recommended by William)
							/*
							robot->rotationInWorld(x_ori, "ra_link6");
							posori_task_handL->_desired_orientation = x_ori;
							robot->rotationInWorld(x_ori, "la_link6");
							posori_task_handR->_desired_orientation = x_ori;
							*/
						}
					}
					break;

				case DODGE_INIT:

					// Define dodge posture //
					q_desired = q_init_desired;
					dodge_posture(q_desired);

					// Generate command torques
					joint_task->_kp = kp_joint+100;
					joint_task->_kv = kv_joint-50;

					joint_task->_desired_position = q_desired;
					joint_task->updateTaskModel(N_prec_feet);
					joint_task->computeTorques(joint_task_torques);

					command_torques = posori_task_torques_footR + posori_task_torques_footL + joint_task_torques;

					state = DODGE;
					cout << "Dodge" << endl;
					break;

				case DODGE:
					// timer
					state_time ++;

					// update the models
					if(counter % nsUpdateFrequency == 0){
						joint_task->updateTaskModel(N_prec_feet);
					}

					joint_task->computeTorques(joint_task_torques);
					command_torques = posori_task_torques_footR + posori_task_torques_footL + joint_task_torques;

					// Checking distance to goal position did not work, so we're just timing out of the dodge
										// cout << (x_pos_bag_end - x_pos).squaredNorm() << endl;
					if(state_time > state_timeout){
						state = NEUTRAL_INIT;
						joint_task->_kp = kp_joint;
						joint_task->_kv = kv_joint;
					}
					break;

				case CROSS_INIT:
					// cout << x_pos_bag_cm.transpose() << " " << x_pos_rh.transpose() << endl;

					// Update posori task
					posori_task_handR->_desired_position = current_desired_pos;

					// Define cross posture
					q_desired = q_init_desired;
					cross_posture(q_desired);
					joint_task->_desired_position = q_desired;

					posori_task_handR->updateTaskModel(N_prec_feet);
					posori_task_handR->computeTorques(posori_task_torques_handR);

					N_prec = posori_task_handR->_N;
					joint_task->updateTaskModel(N_prec);
					joint_task->computeTorques(joint_task_torques);

					command_torques = posori_task_torques_footR + posori_task_torques_footL + posori_task_torques_handR + joint_task_torques;

					cout << "Cross" << endl;
					state = CROSS;
					break;

				case CROSS:
					// update the models
					if(counter % nsUpdateFrequency == 0){
						posori_task_handR->_desired_position = current_desired_pos;
						posori_task_handR->updateTaskModel(N_prec_feet);
						N_prec = posori_task_handR->_N;
						joint_task->updateTaskModel(N_prec);
					}

					posori_task_handR->computeTorques(posori_task_torques_handR);
					joint_task->computeTorques(joint_task_torques);

					command_torques = posori_task_torques_footR + posori_task_torques_footL + posori_task_torques_handR + joint_task_torques;

					// cout << (x_pos_bag_cm - x_pos_rh).squaredNorm() << endl;
					if ((posori_task_handR->_desired_position - x_pos_rh).squaredNorm() < 0.02){
						joint_task->_kp = kp_joint+100; // increase gains to recover faster
						joint_task->_kv = kv_joint-50;
						state = NEUTRAL_INIT;
						state_time = 0;
					}
					break;

				case JAB_INIT:
					// cout << x_pos_bag_cm.transpose() << " " << x_pos_lh.transpose() << endl;

					// Update posori task
					posori_task_handL->_desired_position = current_desired_pos;
					// save desired position

					// Define cross posture
					q_desired = q_init_desired;
					jab_posture(q_desired);
					joint_task->_desired_position = q_desired;

					// posori_task_handL->updateTaskModel(taskProjection * N_prec);
					posori_task_handL->updateTaskModel(N_prec_feet);
					posori_task_handL->computeTorques(posori_task_torques_handL);

					N_prec = posori_task_handL->_N;
					// joint_task->updateTaskModel(jointTaskProjection * N_prec);
					joint_task->updateTaskModel(N_prec);
					joint_task->computeTorques(joint_task_torques);

					command_torques = posori_task_torques_footR + posori_task_torques_footL + posori_task_torques_handL + joint_task_torques;

					cout << "Jab" << endl;
					state = JAB;
					break;

				case JAB:
					//update the models with set frequency
					if(counter % nsUpdateFrequency == 0){
						//cout << "rows: " << N_prec_feet.rows() << " columns: " << N_prec_feet.cols() << endl;
						posori_task_handL->_desired_position = current_desired_pos;
						posori_task_handL->updateTaskModel(N_prec_feet);
						N_prec = posori_task_handL->_N;
						//cout << "N_prec rows: " << N_prec.rows() << " columns: " << N_prec.cols() << endl;
						joint_task->updateTaskModel(N_prec);
					}

					posori_task_handL->computeTorques(posori_task_torques_handL);
					joint_task->computeTorques(joint_task_torques);

					command_torques = posori_task_torques_footR + posori_task_torques_footL + posori_task_torques_handL + joint_task_torques;

					// cout << (x_pos_bag_cm- x_pos_lh).squaredNorm() << endl;
					if ((posori_task_handL->_desired_position - x_pos_lh).squaredNorm() < 0.02){
						joint_task->_kp = kp_joint+100; // Increase gains to recover faster
						joint_task->_kv = kv_joint-50;
						state = NEUTRAL_INIT;
						state_time = 0;
					}
					break;

			}

			// test tracking foot position with only joint task torques to choose posori targets
			Vector3d x_pos_rf;
			robot->positionInWorld(x_pos_rf, "RL_foot", Vector3d(0,0,0));
			Matrix3d x_ori_rf;
			robot->rotationInWorld(x_ori_rf, "RL_foot");

			Vector3d x_pos_lf;
			robot->positionInWorld(x_pos_lf, "LL_foot", Vector3d(0,0,0));
			Matrix3d x_ori_lf;
			robot->rotationInWorld(x_ori_lf, "LL_foot");

			redis_client.setEigenMatrixJSON(OPERATIONAL_POSITION_RF, x_pos_rf.transpose());
			redis_client.setEigenMatrixJSON(OPERATIONAL_POSITION_LF, x_pos_lf.transpose());
			redis_client.setEigenMatrixJSON(OPERATIONAL_ROTATION_RF, x_ori_rf.transpose());
			redis_client.setEigenMatrixJSON(OPERATIONAL_ROTATION_LF, x_ori_lf.transpose());

			// std::cout << "Z actuation position : " << q_desired[2] << "\n";

			// send to redis
			redis_client.setEigenMatrixJSON(JOINT_TORQUES_COMMANDED_KEY, command_torques);

			// ask for next simulation loop
			fSimulationLoopDone = false;
			redis_client.set(SIMULATION_LOOP_DONE_KEY, bool_to_string(fSimulationLoopDone));

			//increment
			++counter;
		}

		// controller loop is done
		fControllerLoopDone = true;
		redis_client.set(CONTROLLER_LOOP_DONE_KEY, bool_to_string(fControllerLoopDone));

	}

	command_torques.setZero();
	redis_client.setEigenMatrixJSON(JOINT_TORQUES_COMMANDED_KEY, command_torques);

	// controller loop is turned off
	fControllerLoopDone = false;
	redis_client.set(CONTROLLER_LOOP_DONE_KEY, bool_to_string(fControllerLoopDone));

	double end_time = timer.elapsedTime();
  std::cout << "\n";
  std::cout << "Controller Loop run time  : " << end_time << " seconds\n";
  std::cout << "Controller Loop updates   : " << counter << "\n";

	return 0;
}

//------------------------------- Functions -------------------------------//
//------------------------------- Functions -------------------------------//
//------------------------------- Functions -------------------------------//

void orthodox_posture(VectorXd& q_desired) {

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
	q_desired[22] = 3*M_PI/4;
	q_desired[23] = 0;
	// q_desired[24] = 0; // Fixed this joint

	// Left Arm
	q_desired[24] = M_PI/6;
	q_desired[25] = M_PI/6;
	q_desired[26] = 0;
	q_desired[27] = 3*M_PI/4;
	q_desired[28] = 0;
	// q_desired[30] = 0;

	// Head
	q_desired[29] = M_PI/6;
}

void dodge_posture(VectorXd& q_desired) {
	orthodox_posture(q_desired);

	// Overactuation
	q_desired[2] = -0.135069;
	q_desired[4] = -M_PI/4;

	// Right Leg
	q_desired[6] = -M_PI/16;
	q_desired[7] = M_PI/4; // Previously 0
	q_desired[8] = 0;
	q_desired[9] = M_PI/4;
	q_desired[10] = M_PI/16;
	q_desired[11] = -M_PI/4;

	// Left Leg
	q_desired[12] = M_PI/16;
	q_desired[13] = 0; // Previously M_PI/4
	q_desired[14] = 0;
	q_desired[15] = M_PI/4;
	q_desired[16] = -M_PI/16;
	q_desired[17] = 0;

	// Trunk
	q_desired[18] = 0;

}

void cross_posture(VectorXd& q_desired) {

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
	q_desired[18] = M_PI/12;

	// Right Arm
	q_desired[19] = M_PI/2;
	q_desired[20] = M_PI/6;
	q_desired[21] = -M_PI/3;
	q_desired[22] = M_PI/12;
	q_desired[23] = 0;
	// q_desired[24] = 0; // fixed this joint

	// Left Arm
	q_desired[24] = M_PI/6;
	q_desired[25] = M_PI/6;
	q_desired[26] = 0;
	q_desired[27] = 3*M_PI/4;;
	q_desired[28] = 0;
	// q_desired[30] = 0; // fixed this joint

	// Head
	q_desired[29] = M_PI/6;

}

void jab_posture(VectorXd& q_desired) {

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
	q_desired[18] = -M_PI/4;

	// Right Arm
	q_desired[19] = M_PI/6;
	q_desired[20] = M_PI/6;
	q_desired[22] = 0;
	q_desired[22] = 3*M_PI/4;;
	q_desired[23] = 0;
	// q_desired[24] = 0; // fixed this joint

	// Left Arm
	q_desired[24] = M_PI/2;
	q_desired[25] = M_PI/6; // M_PI/4;
	q_desired[26] = -M_PI/2;
	q_desired[27] = M_PI/12; // 0
	q_desired[28] = 0;
	// q_desired[30] = M_PI/6; // fixed this joint

	// Head
	q_desired[29] = M_PI/6;
}


bool string_to_bool(const std::string& x) {
  assert(x == "false" || x == "true");
  return x == "true";
}

inline const char * const bool_to_string(bool b)
{
  return b ? "true" : "false";
}
