#include <GL/glew.h>
#include "Sai2Model.h"
#include "Sai2Graphics.h"
#include "Sai2Simulation.h"
#include <dynamics3d.h>
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include <GLFW/glfw3.h>  // must be loaded after loading opengl/glew
#include "uiforce/UIForceWidget.h"  // used for right-click drag interaction in window
#include <random>  // used for white-noise generation
#include "force_sensor/ForceSensorSim.h"  // references src folder in sai2-common directory
#include "force_sensor/ForceSensorDisplay.h"

#include <iostream>
#include <string>

#include <signal.h>
bool fSimulationRunning = false;
void sighandler(int){fSimulationRunning = false;}
bool fSimulationLoopDone = false;
bool fControllerLoopDone = true; // initialize as true for first loop

using namespace std;
using namespace Eigen;

const int timeDilationFactor = 10;

const string world_file = "./resources/world.urdf";
const string robot_file = "./resources/toro.urdf";
const string bag_file = "./resources/punching_bag.urdf";
const string env_file = "./resources/environment.urdf";
const string robot_name = "DLR_TORO";
const string bag_name = "punching_bag";
// const string env_name = "env";
const string camera_name = "camera_fixed";

// redis keys:
// - write:
const std::string JOINT_ANGLES_KEY = "sai2::cs225a::project::sensors::q";
const std::string JOINT_VELOCITIES_KEY = "sai2::cs225a::project::sensors::dq";
const std::string PUNCHING_BAG_ANGLES_KEY = "sai2::cs225a::project::sensors::bag";
// - read
const std::string JOINT_TORQUES_COMMANDED_KEY = "sai2::cs225a::project::actuators::fgc";
// - read + write
const std::string SIMULATION_LOOP_DONE_KEY = "cs225a::simulation::done";
const std::string CONTROLLER_LOOP_DONE_KEY = "cs225a::controller::done";

RedisClient redis_client;

// functions for converting bool <--> string
bool string_to_bool(const std::string& x);
inline const char * const bool_to_string(bool b);

// simulation function prototype
void simulation(Sai2Model::Sai2Model* robot, Sai2Model::Sai2Model* bag, Simulation::Sai2Simulation* sim, UIForceWidget *ui_force_widget);

// callback to print glfw errors
void glfwError(int error, const char* description);

// callback to print glew errors
bool glewInitialize();

// callback when a key is pressed
void keySelect(GLFWwindow* window, int key, int scancode, int action, int mods);

// callback when a mouse button is pressed
void mouseClick(GLFWwindow* window, int button, int action, int mods);

// flags for scene camera movement
bool fTransXp = false;
bool fTransXn = false;
bool fTransYp = false;
bool fTransYn = false;
bool fTransZp = false;
bool fTransZn = false;
bool fRotPanTilt = false;
bool fRobotLinkSelect = false;

int main() {
	cout << "Loading URDF world model file: " << world_file << endl;

	// start redis client
	redis_client = RedisClient();
	redis_client.connect();

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	// load graphics scene
	auto graphics = new Sai2Graphics::Sai2Graphics(world_file, true);
	Eigen::Vector3d camera_pos, camera_lookat, camera_vertical;
	graphics->getCameraPose(camera_name, camera_pos, camera_vertical, camera_lookat);

	// load robots
	auto robot = new Sai2Model::Sai2Model(robot_file, false);
	Matrix3d R_world_bag;
	R_world_bag = AngleAxisd(M_PI/2, Vector3d::UnitZ())
								* AngleAxisd(0.0, Vector3d::UnitY())
								* AngleAxisd(M_PI/2, Vector3d::UnitX());
	Affine3d T_world_bag = Affine3d::Identity();
	T_world_bag.translation() = Vector3d(0.8, 0, 0.82);
	T_world_bag.linear() = R_world_bag;

	auto bag = new Sai2Model::Sai2Model(bag_file, false, T_world_bag);
	auto env = new Sai2Model::Sai2Model(env_file, false);
	env->updateKinematics();

	// load simulation world
	auto sim = new Simulation::Sai2Simulation(world_file, false);
	sim->setCollisionRestitution(0);
	sim->setCoeffFrictionStatic(0.6);

	// read joint positions, velocities, update model
	sim->getJointPositions(robot_name, robot->_q);
	sim->getJointVelocities(robot_name, robot->_dq);
	robot->updateKinematics();

	bag->_q = Vector3d(0,0,-M_PI/12);
	sim->setJointPositions(bag_name, bag->_q);
	sim->getJointVelocities(bag_name, bag->_dq);
	bag->updateKinematics();

	/*------- Set up visualization -------*/
	// set up error callback
	glfwSetErrorCallback(glfwError);

	// initialize GLFW
	glfwInit();

	// retrieve resolution of computer display and position window accordingly
	GLFWmonitor* primary = glfwGetPrimaryMonitor();
	const GLFWvidmode* mode = glfwGetVideoMode(primary);

	// information about computer screen and GLUT display window
	int screenW = mode->width;
	int screenH = mode->height;
	int windowW = 0.8 * screenH;
	int windowH = 0.7 * screenH;
	int windowPosY = (screenH - windowH) / 2;
	int windowPosX = windowPosY;

	// create window and make it current
	glfwWindowHint(GLFW_VISIBLE, 0);
	GLFWwindow* window = glfwCreateWindow(windowW, windowH, "SAI2.0 - PandaApplications", NULL, NULL);
	glfwSetWindowPos(window, windowPosX, windowPosY);
	glfwShowWindow(window);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	// set callbacks
	glfwSetKeyCallback(window, keySelect);
	glfwSetMouseButtonCallback(window, mouseClick);

	// init click force widget
	auto ui_force_widget = new UIForceWidget(robot_name, robot, graphics);
	ui_force_widget->setEnable(false);

	// cache variables
	double last_cursorx, last_cursory;

	// init redis client values
	redis_client.setEigenMatrixJSON(JOINT_ANGLES_KEY, robot->_q);
	redis_client.setEigenMatrixJSON(JOINT_VELOCITIES_KEY, robot->_dq);
	redis_client.setEigenMatrixJSON(PUNCHING_BAG_ANGLES_KEY, bag->_q);
	redis_client.set(SIMULATION_LOOP_DONE_KEY, bool_to_string(fSimulationLoopDone));
	redis_client.set(CONTROLLER_LOOP_DONE_KEY, bool_to_string(fControllerLoopDone));

	// initialize glew
	glewInitialize();

	fSimulationRunning = true;
	thread sim_thread(simulation, robot, bag, sim, ui_force_widget);

	// while window is open:
	while (!glfwWindowShouldClose(window) && fSimulationRunning)
	{
		// update graphics. this automatically waits for the correct amount of time
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		graphics->updateGraphics(robot_name, robot);
		graphics->updateGraphics(bag_name, bag);
		graphics->render(camera_name, width, height);

		// swap buffers
		glfwSwapBuffers(window);

		// wait until all GL commands are completed
		glFinish();

		// check for any OpenGL errors
		GLenum err;
		err = glGetError();
		assert(err == GL_NO_ERROR);

		// poll for events
		glfwPollEvents();

		// move scene camera as required
		// graphics->getCameraPose(camera_name, camera_pos, camera_vertical, camera_lookat);
		Eigen::Vector3d cam_depth_axis;
		cam_depth_axis = camera_lookat - camera_pos;
		cam_depth_axis.normalize();
		Eigen::Vector3d cam_up_axis;
		// cam_up_axis = camera_vertical;
		// cam_up_axis.normalize();
		cam_up_axis << 0.0, 0.0, 1.0; //TODO: there might be a better way to do this
		Eigen::Vector3d cam_roll_axis = (camera_lookat - camera_pos).cross(cam_up_axis);
		cam_roll_axis.normalize();
		Eigen::Vector3d cam_lookat_axis = camera_lookat;
		cam_lookat_axis.normalize();
		if (fTransXp) {
			camera_pos = camera_pos + 0.05*cam_roll_axis;
			camera_lookat = camera_lookat + 0.05*cam_roll_axis;
		}
		if (fTransXn) {
			camera_pos = camera_pos - 0.05*cam_roll_axis;
			camera_lookat = camera_lookat - 0.05*cam_roll_axis;
		}
		if (fTransYp) {
			// camera_pos = camera_pos + 0.05*cam_lookat_axis;
			camera_pos = camera_pos + 0.05*cam_up_axis;
			camera_lookat = camera_lookat + 0.05*cam_up_axis;
		}
		if (fTransYn) {
			// camera_pos = camera_pos - 0.05*cam_lookat_axis;
			camera_pos = camera_pos - 0.05*cam_up_axis;
			camera_lookat = camera_lookat - 0.05*cam_up_axis;
		}
		if (fTransZp) {
			camera_pos = camera_pos + 0.1*cam_depth_axis;
			camera_lookat = camera_lookat + 0.1*cam_depth_axis;
		}
		if (fTransZn) {
			camera_pos = camera_pos - 0.1*cam_depth_axis;
			camera_lookat = camera_lookat - 0.1*cam_depth_axis;
		}
		if (fRotPanTilt) {
			// get current cursor position
			double cursorx, cursory;
			glfwGetCursorPos(window, &cursorx, &cursory);
			//TODO: might need to re-scale from screen units to physical units
			double compass = 0.006*(cursorx - last_cursorx);
			double azimuth = 0.006*(cursory - last_cursory);
			double radius = (camera_pos - camera_lookat).norm();
			Eigen::Matrix3d m_tilt; m_tilt = Eigen::AngleAxisd(azimuth, -cam_roll_axis);
			camera_pos = camera_lookat + m_tilt*(camera_pos - camera_lookat);
			Eigen::Matrix3d m_pan; m_pan = Eigen::AngleAxisd(compass, -cam_up_axis);
			camera_pos = camera_lookat + m_pan*(camera_pos - camera_lookat);
		}
		graphics->setCameraPose(camera_name, camera_pos, cam_up_axis, camera_lookat);
		glfwGetCursorPos(window, &last_cursorx, &last_cursory);

		ui_force_widget->setEnable(fRobotLinkSelect);
		if (fRobotLinkSelect)
		{
			double cursorx, cursory;
			int wwidth_scr, wheight_scr;
			int wwidth_pix, wheight_pix;
			std::string ret_link_name;
			Eigen::Vector3d ret_pos;

			// get current cursor position
			glfwGetCursorPos(window, &cursorx, &cursory);

			glfwGetWindowSize(window, &wwidth_scr, &wheight_scr);
			glfwGetFramebufferSize(window, &wwidth_pix, &wheight_pix);

			int viewx = floor(cursorx / wwidth_scr * wwidth_pix);
			int viewy = floor(cursory / wheight_scr * wheight_pix);

			if (cursorx > 0 && cursory > 0)
			{
				ui_force_widget->setInteractionParams(camera_name, viewx, wheight_pix - viewy, wwidth_pix, wheight_pix);
				//TODO: this behavior might be wrong. this will allow the user to click elsewhere in the screen
				// then drag the mouse over a link to start applying a force to it.
			}
		}
	}

	// wait for simulation to finish
	fSimulationRunning = false;
	fSimulationLoopDone = false;
	redis_client.set(SIMULATION_LOOP_DONE_KEY, bool_to_string(fSimulationLoopDone));
	sim_thread.join();

	// destroy context
	glfwDestroyWindow(window);

	// terminate
	glfwTerminate();

	return 0;
}

//------------------------------------------------------------------------------
void simulation(Sai2Model::Sai2Model* robot, Sai2Model::Sai2Model* bag, Simulation::Sai2Simulation* sim, UIForceWidget *ui_force_widget) {

	int dof = robot->dof();
	VectorXd command_torques = VectorXd::Zero(dof);
	VectorXd bag_q = VectorXd::Zero(3);

	redis_client.setEigenMatrixJSON(JOINT_TORQUES_COMMANDED_KEY, command_torques);
	redis_client.setEigenMatrixJSON(PUNCHING_BAG_ANGLES_KEY, bag_q);

	// create a timer
	LoopTimer timer;
	timer.initializeTimer();
	timer.setLoopFrequency(1000);
	double start_time = timer.elapsedTime() / timeDilationFactor ; // secs
	double last_time = timer.elapsedTime(); //secs
	bool fTimerDidSleep = true;

	// init variables
	VectorXd robot_g = VectorXd::Zero(dof);
	VectorXd bag_g = VectorXd::Zero(3);

	Eigen::Vector3d ui_force;
	ui_force.setZero();

	Eigen::VectorXd ui_force_command_torques;
	ui_force_command_torques.setZero();

	// setup redis client data container for pipeset (batch write)
	std::vector<std::pair<std::string, std::string>> redis_data(6);  // set with the number of keys to write

	unsigned long long counter = 0;

	fSimulationRunning = true;
	while (fSimulationRunning) {

		if (fControllerLoopDone || fRobotLinkSelect) {

			// get gravity torques
			robot->gravityVector(robot_g);
			bag->gravityVector(bag_g);

			// read arm torques from redis and apply to simulated robot
			if (fControllerLoopDone) {
				command_torques = redis_client.getEigenMatrixJSON(JOINT_TORQUES_COMMANDED_KEY);
			}
			else {
				command_torques.setZero();
			}

			ui_force_widget->getUIForce(ui_force);
			ui_force_widget->getUIJointTorques(ui_force_command_torques);

			if (fRobotLinkSelect) {
				sim->setJointTorques(robot_name, command_torques + ui_force_command_torques + robot_g);
				sim->setJointTorques(bag_name, ui_force_command_torques -30*bag->_dq);
			}
			else {
				sim->setJointTorques(robot_name, command_torques + robot_g);
				sim->setJointTorques(bag_name, ui_force_command_torques -30*bag->_dq);
			}

			// integrate forward
			double curr_time = timer.elapsedTime();
			double loop_dt = curr_time - last_time;
			sim->integrate(0.001); // integrate at 1 kHz

			// read joint positions, velocities, update model
			sim->getJointPositions(robot_name, robot->_q);
			sim->getJointVelocities(robot_name, robot->_dq);
			robot->updateModel();
			sim->getJointPositions(bag_name, bag->_q);
			sim->getJointVelocities(bag_name, bag->_dq);
			bag->updateModel();

			// simulation loop is done
	    fSimulationLoopDone = true;

      // ask for next control loop
      fControllerLoopDone = false;

			// write new robot state to redis
			redis_data.at(0) = std::pair<string, string>(JOINT_TORQUES_COMMANDED_KEY, redis_client.encodeEigenMatrixJSON(command_torques));
			redis_data.at(1) = std::pair<string, string>(JOINT_ANGLES_KEY, redis_client.encodeEigenMatrixJSON(robot->_q));
			redis_data.at(2) = std::pair<string, string>(JOINT_VELOCITIES_KEY, redis_client.encodeEigenMatrixJSON(robot->_dq));
			redis_data.at(3) = std::pair<string, string>(PUNCHING_BAG_ANGLES_KEY, redis_client.encodeEigenMatrixJSON(bag->_q));
			redis_data.at(4) = std::pair<string, string>(SIMULATION_LOOP_DONE_KEY, bool_to_string(fSimulationLoopDone));
			redis_data.at(5) = std::pair<string, string>(CONTROLLER_LOOP_DONE_KEY, bool_to_string(fControllerLoopDone)); // ask for next control loop

			redis_client.pipeset(redis_data);

			// update last time
			last_time = curr_time;

			++counter;
		}

		// read controller state
    fControllerLoopDone = string_to_bool(redis_client.get(CONTROLLER_LOOP_DONE_KEY));

	}

	double end_time = timer.elapsedTime();
	cout << "\n";
	cout << "Simulation Loop run time  : " << end_time << " seconds\n";
	cout << "Simulation Loop updates   : " << counter << "\n";

}

//------------------------------------------------------------------------------

void glfwError(int error, const char* description) {
	cerr << "GLFW Error: " << description << endl;
	exit(1);
}

//------------------------------------------------------------------------------

bool glewInitialize() {
	bool ret = false;
	#ifdef GLEW_VERSION
	if (glewInit() != GLEW_OK) {
		cout << "Failed to initialize GLEW library" << endl;
		cout << glewGetErrorString(ret) << endl;
		glfwTerminate();
	} else {
		ret = true;
	}
	#endif
	return ret;
}

//------------------------------------------------------------------------------

void keySelect(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	bool set = (action != GLFW_RELEASE);
	switch(key) {
		case GLFW_KEY_ESCAPE:
			// exit application
			fSimulationRunning = false;
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		case GLFW_KEY_RIGHT:
			fTransXp = set;
			break;
		case GLFW_KEY_LEFT:
			fTransXn = set;
			break;
		case GLFW_KEY_UP:
			fTransYp = set;
			break;
		case GLFW_KEY_DOWN:
			fTransYn = set;
			break;
		case GLFW_KEY_A:
			fTransZp = set;
			break;
		case GLFW_KEY_Z:
			fTransZn = set;
			break;
		default:
			break;
	}
}

//------------------------------------------------------------------------------

void mouseClick(GLFWwindow* window, int button, int action, int mods) {
	bool set = (action != GLFW_RELEASE);
	//TODO: mouse interaction with robot
	switch (button) {
		// left click pans and tilts
		case GLFW_MOUSE_BUTTON_LEFT:
			fRotPanTilt = set;
			// NOTE: the code below is recommended but doesn't work well
			// if (fRotPanTilt) {
			// 	// lock cursor
			// 	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			// } else {
			// 	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			// }
			break;
		// if right click: don't handle. this is for menu selection
		case GLFW_MOUSE_BUTTON_RIGHT:
			fRobotLinkSelect = set;
			break;
		// if middle click: don't handle. doesn't work well on laptops
		case GLFW_MOUSE_BUTTON_MIDDLE:
			break;
		default:
			break;
	}
}

bool string_to_bool(const std::string& x) {
  assert(x == "false" || x == "true");
  return x == "true";
}

inline const char * const bool_to_string(bool b)
{
  return b ? "true" : "false";
}
