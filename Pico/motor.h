#ifndef MOTOR_H
#define MOTOR_H

class Motor {
	public:
		int dir1;
		int dir2;
		int speed;
		
		Motor(int d1, int d2, int sp, pwm_config cfg)
		{
			// Assign pin values
			dir1 = d1;
			dir2 = d2;
			speed = sp;
			
			// Initalize direction pins as GPIO outputs
			gpio_init(dir1);
			gpio_set_dir(dir1, GPIO_OUT);
			gpio_init(dir2);
			gpio_set_dir(dir2, GPIO_OUT);
			
			// Initialize speed pin as a PWM output
			pwm_init(pwm_gpio_to_slice_num(speed), &cfg, true);
			gpio_set_function(speed, GPIO_FUNC_PWM);
		}
		
		// Change motor speed and direction
		//	Input should be a value in terms of degrees
		void setSpeed(float value)
		{
			//float degree = rad2deg(value);
			
			// Change direction based on value
			//   e.g. positive -> left, negative -> right
			int toSend = 0;
			double threshold = 12;
			if(value < -threshold)
			{
				gpio_put(dir1, 1);
				gpio_put(dir2, 0);
				toSend = 255;
			}
			else if(value > threshold)
			{
				gpio_put(dir1, 0);
				gpio_put(dir2, 1);
				toSend = 255;
			}
			
			// Set motor speed
			pwm_set_gpio_level(speed, toSend);
		}
};

#endif