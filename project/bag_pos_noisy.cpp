void bag_pos_noisy(Vector3d& x_pos_bag_noisy, const double noise_magnitude);

//to use, just use x_pos_bag_noisy in place of x_pos bag. We can vary noise_magnitude to make the noise bigger or smaller

//function
void bag_pos_noisy(Vector3d& x_pos_bag_noisy, const double noise_magnitude){
	//get real position
	Matrix3d R_world_bag;
	R_world_bag = AngleAxisd(M_PI/2, Vector3d::UnitX())
								* AngleAxisd(0.0, Vector3d::UnitY())
								* AngleAxisd(M_PI/2, Vector3d::UnitZ());
	Affine3d T_world_bag = Affine3d::Identity();
	T_world_bag.translation() = Vector3d(0.75, 0, 0.82);
	T_world_bag.linear() = R_world_bag;
	auto bag = new Sai2Model::Sai2Model(bag_file, false, T_world_bag);
	//bag->updateModel();
	Vector3d bag_cm = R_world_bag*Vector3d(0, 0.8, 0);

	Vector3d x_pos_bag;
	bag->positionInWorld(x_pos_bag, "bag", bag_cm);

	// setup white noise generator
    const double mean = 0.0;
    const double stddev = noise_magnitude;  // tune based on your system 
    std::default_random_engine generator;
    std::normal_distribution<double> dist(mean, stddev);

	//get + add noise
	x_pos_bag_noisy[0] = x_pos_bag[0] + dist(generator); 
	x_pos_bag_noisy[1] =x_pos_bag[1] + dist(generator); 
	x_pos_bag_noisy[2] =x_pos_bag[2] + dist(generator); 
	
}