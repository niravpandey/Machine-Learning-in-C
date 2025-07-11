#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define CAP 10     // initial capacity of array
#define LR  0.001  // learning rate of linear regression algorithm
#define ITR 10000  // number of iterations

// point struct to store x and y values
typedef struct{
    double x;
    double y;
} point_t;

double distance(point_t a, point_t b);
point_t* read_points(int* count);

int main() {
    int m;
    point_t* points = read_points(&m);

    double theta_0 = 0.0, theta_1 = 0.0;

    for (int j = 0; j < ITR; j++) {
        double grad_0 = 0.0;
        double grad_1 = 0.0;

        for (int i = 0; i < m; i++) {
            double h = theta_1 * points[i].x + theta_0;
            double error = h - points[i].y;

            grad_0 += error;                    
            grad_1 += error * points[i].x;      
        }

        grad_0 /= m;
        grad_1 /= m;

        theta_0 -= LR * grad_0;
        theta_1 -= LR * grad_1;
    }

    printf("\nFinal result:\n");
    printf("theta_0: %.2f\n", theta_0);
    printf("theta_1: %.2f\n", theta_1);

    free(points);
    return 0;
}


point_t* read_points(int* count){
    int cap = CAP;
    int i = 0; 
    point_t* arr = malloc(cap*sizeof(point_t));
    if (!arr){
        fprintf(stderr, "malloc() failed!\n");
        exit(1);
    }
    while(scanf("%lf %lf", &arr[i].x, &arr[i].y)==2){
        i++;
        if (i>= cap){
            cap *=2;
            point_t* temp = realloc(arr,cap*sizeof(point_t));
            if(!temp){
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

double distance(point_t a, point_t b){
    return sqrt(pow((a.x-b.x),2)+pow((a.y-b.y),2));
}

