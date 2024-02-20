#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

using  namespace std;

class Vec3 {
public:
    double x, y, z;
    // Constructors
    //Vec3() : x(0), y(0), z(0) {}
    Vec3(double x = 0.0, double y = 0.0, double z = 0.0) : x(x), y(y), z(z) {}

    // Define equality operator
    bool operator==(const Vec3& other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    // Define inequality operator
    bool operator!=(const Vec3& other) const {
        return !(*this == other);
    }

    // Overload unary minus operator
    Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }

    // Overload subtraction operator
    Vec3 operator-(const Vec3& other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }

    // Overload division operator
    Vec3 operator/(double scalar) const {
        return Vec3(x / scalar, y / scalar, z / scalar);
    }

    // Overload multiplication operator for scalar and Vec3
    Vec3 operator*(double scalar) const {
        return Vec3(x * scalar, y * scalar, z * scalar);
    }

     // Overload multiplication operator for scalar and Vec3
    friend Vec3 operator*(double scalar, const Vec3& vec) {
        return vec * scalar;
    }

    // Overload addition operator for Vec3
    Vec3 operator+(const Vec3& other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }

     // Overload compound addition operator for Vec3
    Vec3& operator+=(const Vec3& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

     // Define dot product function
    double dot(const Vec3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }  
};
// Define Material struct
struct Material {
    Vec3 diffuseColor; // Diffuse color components
    Vec3 specularColor; // Specular color components
    double ambientCoefficient; // Ambient coefficient
    double diffuseCoefficient; // Diffuse coefficient
    double specularCoefficient; // Specular coefficient
    double specularExponent; // Specular exponent
};

double dot_product(const Vec3& v1, const Vec3& v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

// Define Sphere struct
struct Sphere {
    Vec3 center;
    double radius;
    Material material; // Include Material member

    Sphere(const Vec3& center, double radius, const Material& material) 
        : center(center), radius(radius), material(material) {}
};


struct Light {
    Vec3 position;
    int type;
    double intensity;
};

// Function for cross product of vectors
Vec3 crossProduct(Vec3 a, Vec3 b) {
  Vec3 result;

  result.x = a.y*b.z - a.z*b.y;
  result.y = a.z*b.x - a.x*b.z;
  result.z = a.x*b.y - a.y*b.x;

  return result;
}

// Function to calculate the squared distance between two points
double distance_squared(const Vec3& point1, const Vec3& point2) {
    double dx = point1.x - point2.x;
    double dy = point1.y - point2.y;
    double dz = point1.z - point2.z;
    return dx * dx + dy * dy + dz * dz;
}

// Function for Normalization of vector
Vec3 normalize(Vec3 v) {
  double mag = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
  Vec3 normalized = {v.x/mag, v.y/mag, v.z/mag};
  return normalized;
}

// Class for ray
class ray {
  public:
    
    ray() {}

    ray(const Vec3& origin, const Vec3& direction) : eye(origin), viewDir(direction) {}

    Vec3 getOrigin() const { return eye; }
    Vec3 getDirection() const { return viewDir; }

    Vec3 at(double t) const {
        return eye + viewDir * t;
    }

  private:
    Vec3 eye;
    Vec3 viewDir;
};

// Function to find the intersection of a ray with a sphere
bool raySphereIntersect(const ray& r, const Sphere& sphere, double& t0, double& t1) {

  Vec3 oc = r.getOrigin() - sphere.center;
  
  double a = r.getDirection().dot(r.getDirection());
  double b = 2.0 * oc.dot(r.getDirection());
  double c = oc.dot(oc) - sphere.radius*sphere.radius;

    double discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        return false;
    }

    double sqrtDisc = sqrt(discriminant);
    t0 = (-b - sqrtDisc) / (2 * a);
    t1 = (-b + sqrtDisc) / (2 * a);

    return true;
}

// Function to check intersection with a sphere and return the smallest positive t
bool intersect(const ray& r, const Sphere& sphere, double& t) {
  
  double t0, t1;
  if (raySphereIntersect(r, sphere, t0, t1)) {
    t = min(t0, t1);
    // for debugging  purposes 
    //cout << "Intersecting sphere: " << sphere.center.x << " " << sphere.center.y << " " << sphere.center.z << " with t: " << t << endl;
    return t > 0;
  }

  return false;
}
  // Function to check shadow from one object to another
bool shadow_ray_intersect(const Vec3& intersection_point, const std::vector<Light>& lights, const std::vector<Sphere>& spheres, int index) {
    // Iterate over each light source
    for (const Light& light : lights) {
        double min_t = INFINITY;

        // Calculate shadow ray direction
        Vec3 shadow_ray_direction = normalize(light.position - intersection_point);

        // Iterate over all spheres in the scene
        for (size_t i = 0; i < spheres.size(); ++i) {
            if (i == index) continue; // Skip the object for which the illumination is being computed

            const Sphere& sphere = spheres[i];

            // Offset origin slightly to avoid self-intersections
            Vec3 offset_origin = intersection_point + shadow_ray_direction * 0.001;

            ray shadow_ray(offset_origin, shadow_ray_direction);

            double t;
            if (intersect(shadow_ray, sphere, t) && t > 0 && t < min_t) {
                // Intersection found, object is blocking the light
                min_t = t; // Update min_t to consider the closest blocking object
            }
        }

        // Check if the closest blocking object is behind the light source
        if (min_t < distance_squared(intersection_point, light.position)) {
            return true; // Intersection found, point is in shadow for at least one light source
        }
    }

    // No intersection found for any light source, point is not in shadow
    return false;
}


// Function to calculate Phong shading for a given intersection point
Vec3 blinnPhongShading(const Vec3& intersection_point, const Vec3& normal, const Vec3& V_vector, const std::vector<Light>& lights, const Material& material, const std::vector<Sphere>& spheres,int index) {
    // Initialize final color
    Vec3 final_color(0.0, 0.0, 0.0);

    // Iterate over all lights
    for (const Light& light : lights) {
        Vec3 lightDir;
        if (light.type == 1) {
            // Point light
            lightDir = normalize(light.position - intersection_point);
        } else {
            // Directional light
            lightDir = -normalize(light.position);
        }

        // Check for shadows
        bool in_shadow = shadow_ray_intersect(intersection_point, lights, spheres, index);


        if (!in_shadow) {
            // Diffuse component
            double diffuseIntensity = std::max(0.0, normal.dot(lightDir));
            Vec3 diffuse = material.diffuseColor * material.diffuseCoefficient * diffuseIntensity;

            // Specular component
            Vec3 halfwayDir = normalize(V_vector + lightDir);
            double specularIntensity = std::pow(std::max(0.0, normal.dot(halfwayDir)), material.specularExponent);
            Vec3 specular = material.specularColor * material.specularCoefficient * specularIntensity;

            // Add shading contribution from this light
            final_color += (diffuse + specular) * light.intensity;
        }
    }

    // Add ambient component
    final_color += material.diffuseColor * material.ambientCoefficient;

    // Clamp final color values to [0, 1]
    final_color.x = std::max(0.0, std::min(1.0, final_color.x));
    final_color.y = std::max(0.0, std::min(1.0, final_color.y));
    final_color.z = std::max(0.0, std::min(1.0, final_color.z));

    return final_color;
}

using color = Vec3;
// Functin to define the colors of ray intersection with sphere.
color shade_ray(const ray& r, const std::vector<Sphere>& spheres, const std::vector<Material>& materials, const std::vector<Light>& lights, const Vec3& bkgColor, const Vec3& eye) {
    double min_t = INFINITY;
    Material defaultMaterial;
    defaultMaterial.diffuseColor = Vec3(0.0, 0.0, 0.0);  // Diffuse color (black)
    defaultMaterial.specularColor = Vec3(0.0, 0.0, 0.0); // Specular color (black)
    defaultMaterial.ambientCoefficient = 0.2;  // Ambient coefficient
    defaultMaterial.diffuseCoefficient = 0.6;  // Diffuse coefficient
    defaultMaterial.specularCoefficient = 0.2; // Specular coefficient
    defaultMaterial.specularExponent = 10;    // Specular exponent

    color final_color = bkgColor;
    int index=0;
    for (const Sphere& sphere : spheres) {
        double t;
        if (intersect(r, sphere, t) && t < min_t) {
            min_t = t;

            // Calculate intersection point
            Vec3 intersection_point = r.at(min_t);

            // Calculate normal at intersection point
            Vec3 normal_temp = (intersection_point - sphere.center) / sphere.radius;
            Vec3 normal = normalize(normal_temp);

            // Calculate view direction
            Vec3 V_vector = normalize(eye - intersection_point);

            // Calculate shading using the Phong model with multiple lights and shadows
            final_color = blinnPhongShading(intersection_point, normal, V_vector, lights, sphere.material, spheres,index);
        }
        index++;
    }

    return final_color;
}

//write each pixel's color to the output image as soon as it is computed
void write_color(ostream &out, Vec3 pixel_color) {
    // Write the translated [0,255] value of each color component.
    out << static_cast<int>(255* pixel_color.x) << ' '
        << static_cast<int>(255* pixel_color.y) << ' '
        << static_cast<int>(255* pixel_color.z) << '\n';
}

// Function to parse scene description from input file
bool parseSceneDescription(const string& filename, Vec3& eye, Vec3& viewDir, Vec3& upDir,
                            double& hfov, int& width, int& height, Vec3& bkgColor,
                            vector<Sphere>& spheres, vector<Material>& materials, vector<Light>& lights) {
  
    ifstream inputFile(filename);
    if (!inputFile) {
        cerr << "Error: Unable to open input file " << filename << endl;
        return false;
    }

    Vec3 defaultColor(1.0, 0.0, 0.0);  // Default color or any color you want
    Material currentMaterial; // Declare currentMaterial variable

    string token;
    while (inputFile >> token) {
        if (token == "eye") {
            inputFile >> eye.x >> eye.y >> eye.z;
        } else if (token == "viewdir") {
            inputFile >> viewDir.x >> viewDir.y >> viewDir.z;
        } else if (token == "updir") {
            inputFile >> upDir.x >> upDir.y >> upDir.z;
        } else if (token == "hfov") {
            inputFile >> hfov;
        } else if (token == "imsize") {
            inputFile >> width >> height;
        } else if (token == "bkgcolor") {
            inputFile >> bkgColor.x >> bkgColor.y >> bkgColor.z;
        } else if (token == "mtlcolor") {
            double Odr, Odg, Odb; // Diffuse color components
            double Osr, Osg, Osb; // Specular color components
            double ka, kd, ks;    // Ambient, diffuse, and specular coefficients
            double n;             // Specular exponent
            inputFile >> Odr >> Odg >> Odb >> Osr >> Osg >> Osb >> ka >> kd >> ks >> n;

            // Create Vec3 objects for material colors
            Vec3 diffuseColor(Odr, Odg, Odb);
            Vec3 specularColor(Osr, Osg, Osb);

            // Update the current material with parsed parameters
            currentMaterial.diffuseColor = diffuseColor;
            currentMaterial.specularColor = specularColor;
            currentMaterial.ambientCoefficient = ka;
            currentMaterial.diffuseCoefficient = kd;
            currentMaterial.specularCoefficient = ks;
            currentMaterial.specularExponent = n;
        } else if (token == "sphere") {
            double Cx, Cy, Cz, r;
            inputFile >> Cx >> Cy >> Cz >> r;
            // Create a new sphere with current material
            Sphere sphere(Vec3(Cx, Cy, Cz), r, currentMaterial);
            spheres.push_back(sphere); // Fix typo here
        } else if (token == "light") {
            Vec3 position;
            double intensity;
            int type; // 1 for point light, 0 for directional light
            inputFile >> position.x >> position.y >> position.z >> type >> intensity;

            // Create a new Light object and add it to the lights vector
            Light light;
            light.position = position;
            light.type = type;
            light.intensity = intensity;

            // Add the light to the lights vector
            lights.push_back(light);
        } else {
            cerr << "Warning: Unknown token '" << token << "' in input file." << endl;
        }
    }
    inputFile.close();
    return true;
}

// Function to calculate viewport corners
void calculateViewportCorners(const Vec3& eye, const Vec3& viewDir, const Vec3& upDir,
                               double hfov, int width, int height, Vec3& ul, Vec3& ur,
                               Vec3& ll, Vec3& lr) {
    // Calculate view parameters
     // u' = viewDir x upDir
    Vec3 uPrime = crossProduct(viewDir, upDir);
    // Normalize uPrime
    Vec3 u = normalize(uPrime);
    // v' = u X viewDir
    Vec3 vPrime = crossProduct(u, viewDir);
      // Normalize vPrime
    Vec3 v = normalize(vPrime);
    Vec3 n = normalize(viewDir);
    Vec3 n1= -n;

    double aspectRatio = static_cast<double>(width) / static_cast<double>(height);
    double viewAngle = hfov * 3.14 / 180;
    double w = width; // Window width
    double h = height; // Window height
    double d = w / (2 * tan(viewAngle / 2)); // Distance to window

    // Calculate viewport corners
    ul = eye + d * n - (w/2.0) * u + (h/2.0) * v;
    ur = eye + d * n + (w/2.0) * u + (h/2.0) * v;
    ll = eye + d * n - (w/2.0) * u - (h/2.0) * v;
    lr = eye + d * n + (w/2.0) * u - (h/2.0) * v;
}

void render_scene(std::ofstream& outputFile, const Vec3& eye, const Vec3& ul, const Vec3& delta_u, const Vec3& delta_v, int width, int height, const std::vector<Sphere>& spheres, const std::vector<Material>& materials, const std::vector<Light>& lights, const Vec3& bkgColor) {
    // PPM standard header
    outputFile << "P3\n";
    outputFile << "# PPM Image\n";
    outputFile << width << " " << height << "\n";
    outputFile << "255\n";

    std::cout << "Before render starts.. " << std::endl;
     // Print positions of lights
    cout << "Positions of lights:" << endl;
    for (const Light& light : lights) {
         cout << "Light position: " << light.position.x << " " << light.position.y << " " << light.position.z << endl;
     }
     
     int counter = 0;
    for (const auto& sph : spheres) {
         cout << "sphere: " << sph.center.x << " " << sph.center.y << " " << sph.center.z 
              << " " << sph.radius << "\n";
         cout << "mtlcolor: " << sph.material.diffuseColor.x << " " 
             << sph.material.diffuseColor.y << " " << sph.material.diffuseColor.z << "\n";
        cout << "--------" << endl;
       counter++;
   }
    cout << "Sphere count: " << counter << endl;

    // Render...
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            auto pixel_center = ul + (delta_u * i) + (delta_v * j);
            auto ray_direction = pixel_center - eye;
            ray_direction = normalize(ray_direction);
            ray r(eye, ray_direction); 

            Vec3 pixel_color = shade_ray(r, spheres, materials, lights, bkgColor, eye);

            write_color(outputFile, pixel_color);
        }
    }

    std::cout << "Image generated successfully." << std::endl;
}



//main function
int main(int argc, char* argv[]) {

  if(argc != 2) {
    cerr << "Usage: " << argv[0] << " <input_file>" << endl;
    return 1;
  }

  string outfile = argv[1];
  outfile.replace(outfile.find(".txt"), 4, ".ppm");

  ofstream outputFile(outfile);
  if(!outputFile) {
    cerr << "Unable to open output file " << outfile << endl;
    return 1;
  }
    
  Vec3 eye, viewDir, upDir;
  double hfov;
  int width, height;
  Vec3 bkgColor, mtlcolor;
  vector<Sphere> spheres;
  vector<Material> materials;
  vector<Light> lights;

    // Parse scene description from input file
    if (!parseSceneDescription(argv[1], eye, viewDir, upDir, hfov, width, height, bkgColor, spheres, materials, lights)) {
        return 1;
    }
  Vec3 ul, ur, ll, lr;
  calculateViewportCorners(eye, viewDir, upDir, hfov, width, height, ul, ur, ll, lr);

  Vec3 delta_u = (ur - ul) / (width - 1);
  Vec3 delta_v = (ll - ul) / (height - 1);

  render_scene(outputFile, eye, ul, delta_u, delta_v, width, height, spheres, materials, lights, bkgColor);

  outputFile.close();

  return 0;
}

