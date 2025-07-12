#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CAP 10        // initial capacity of array
#define LR  0.001     // learning rate of linear regression algorithm
#define EPS 0.000001  // epsilon, an error term to check for convergence
#define N   2         // number of parameters

// point struct to store x and y values
typedef struct {
    double x;
    double y;
} point_t;

// struct to store paramters
typedef struct {
    double theta_0;
    double theta_1;
} params_t;

double distance(point_t a, point_t b);
point_t* read_points(int* count);
params_t gradient_descent(point_t* points, 
    int num_points,
    params_t* input);
void compute_gradients(double* grad_0, 
    double* grad_1,
    point_t* points, 
    int num_points, 
    params_t* input);
void print_result(params_t output);

int main() {
    int num_points  = 0;
    params_t input  = {0, 0};
    point_t* points = read_points(&num_points);
    params_t output = gradient_descent(points, num_points, &input);
    print_result(output);
    free(points);
    return 0;
}

point_t* read_points(int* count) {
    int cap = CAP;
    int i = 0;
    point_t* arr = malloc(cap * sizeof(point_t));
    if (!arr) {
        fprintf(stderr, "malloc() failed!\n");
        exit(1);
    }
    printf("Enter points line-by-line, Cmd+D to end...\n");
    while (scanf("%lf %lf", &arr[i].x, &arr[i].y) == 2) {
        i++;
        if (i >= cap) {
            cap *= 2;
            point_t* temp = realloc(arr, cap * sizeof(point_t));
            if (!temp) {
                fprintf(stderr, "realloc() failed!\n");
                free(arr);
                exit(1);
            }
            arr = temp;
        }
    }

    *count = i;
    return arr;
}

void compute_gradients(double* grad_0, 
    double* grad_1, 
    point_t* points, 
    int num_points, 
    params_t* input) {
    *grad_0 = 0;
    *grad_1 = 0;
    for (int i = 0; i < num_points; i++) {
        double h = input->theta_1 * points[i].x + input->theta_0;
        double error = h - points[i].y;

        *grad_0 += error;
        *grad_1 += error * points[i].x;
    }

    *grad_0 /= num_points;
    *grad_1 /= num_points;
}

params_t gradient_descent(point_t* points,
    int num_points, 
    params_t* input) {
    double grad_0, grad_1;

    // initial gradient computation
    compute_gradients(&grad_0, &grad_1, points, num_points, input);

    while (fabs(grad_0) > EPS || fabs(grad_1) > EPS) {
        // update parameters
        input->theta_0 -= LR * grad_0;
        input->theta_1 -= LR * grad_1;

        // recompute gradients after parameter update
        compute_gradients(&grad_0, &grad_1, points, num_points, input);
    }

    return *input;
}

double distance(point_t a, point_t b) {
    return sqrt(pow((a.x - b.x), 2) + pow((a.y - b.y), 2));
}

void print_result(params_t output){
    printf("\nFinal result:\n");
    printf("-------------\n");
    printf("theta_0: %.2f\n", output.theta_0);
    printf("theta_1: %.2f\n", output.theta_1);
}