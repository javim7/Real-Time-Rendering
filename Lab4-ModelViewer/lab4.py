import glm
import numpy
import pygame
import random
from obj import *
from math import *
from OpenGL.GL import *
from OpenGL.GL.shaders import *

pygame.init()

screen = pygame.display.set_mode(
    (700, 500),
    pygame.OPENGL | pygame.DOUBLEBUF
)


vertex_shader = """
#version 460
layout (location = 0) in vec3 position;
layout (location = 1) in vec3 vertexColor;

uniform mat4 amatrix;

out vec3 ourColor;
out vec2 fragCoord;

void main()
{
    gl_Position = amatrix * vec4(position, 1.0f);
    ourColor = vertexColor;
    fragCoord = gl_Position.xy;

}
"""

fragment_shader = """
#version 460

layout (location = 0) out vec4 fragColor;

uniform vec3 color;


in vec3 ourColor;

void main()
{
    // fragColor = vec4(ourColor, 1.0f);
    fragColor = vec4(color, 1.0f);
}
"""

fragment_shader1 = """

#version 460

layout (location = 0) out vec4 fragColor;
in vec3 ourColor;
in vec2 fragCoord;

uniform float iTime;
#define time iTime
#define resolution iResolution
#define so fract(sin(time)*123.456)

float det=.001, br=0., tub=0., hit=0.;
vec3 pos, sphpos;
mat3 lookat(vec3 dir, vec3 up) {
  vec3 rt=normalize(cross(dir,up));
  return mat3(rt,cross(rt,dir),dir);
}
vec3 path(float t) {
  return vec3(sin(t+cos(t)*.5)*.5,cos(t*.5),t);
}
mat2 rot(float a) {
  float s=sin(a);
  float c=cos(a);
  return mat2(c,s,-s,c);
}
vec3 fractal(vec2 p) {
  p=fract(p*.1);
  float m=1000.;
  for (int i=0; i<7; i++) {
    p=abs(p)/clamp(abs(p.x*p.y),.25,2.)-1.2;
    m=min(m,abs(p.y)+fract(p.x*.3+time*.5+float(i)*.25));
  }
  m=exp(-6.*m);
  return m*vec3(abs(p.x),m,abs(p.y));
}

float coso(vec3 pp) {
  pp*=.7;
  pp.xy*=rot(pp.z*2.);
  pp.xz*=rot(time*2.);
  pp.yz*=rot(time);
  float sph=length(pp)-.04;
  sph-=length(sin(pp*40.))*.05;
  sph=max(sph,-length(pp)+.11);
  float br2=length(pp)-.03;
  br2=min(br2,length(pp.xy)+.005);
  br2=min(br2,length(pp.xz)+.005);
  br2=min(br2,length(pp.yz)+.005);
  br2=max(br2,length(pp)-1.);
  br=min(br2,br);
  float d=min(br,sph);
  return d;
}


float de(vec3 p) {
  hit=0.;
  br=1000.;
  vec3 pp=p-sphpos;
  p.xy-=path(p.z).xy;
  p.xy*=rot(p.z+time*.5);
  float s=sin(p.z*.5+time*.5);
  p.xy*=1.3-s*s*.7;
  
  for(int i=0; i<6; i++) {
    p=abs(p)-.4;
  }
  pos=p;
  tub=-length(p.xy)+.45+sin(p.z*10.)*.1*smoothstep(.4,.5,abs(.5-fract(p.z*.05))*2.);
  float co=coso(pp);
  co=min(co,coso(pp+.7));
  co=min(co,coso(pp-.7));
  float d=min(tub,co);
  if (d==tub) hit=step(fract(.1*length(sin(p*10.))),.05);
  return d*.3;
}

vec3 march(vec3 from, vec3 dir) {
  vec2 uv=vec2(atan(dir.x,dir.y)+time*.5,length(dir.xy)+sin(time*.2));
  vec3 col=fractal(uv);
  float d=0.,td=0.,g=0., ref=0., ltd=0., li=0.;
  vec3 p=from;
  for (int i=0; i<200; i++) {
    p+=dir*d;
    d=de(p);
    if (d<det && ref==0. && hit==1.) {
      vec2 e=vec2(0.,.1);
      vec3 n=normalize(vec3(de(p+e.yxx),de(p+e.xyx),de(p+e.xxy))-de(p));
      p-=dir*d*2.;
      dir=reflect(dir,n);
      ref=1.;
      td=0.;
      ltd=td;
      continue;
    }
    if (d<det || td>5.) break;
    td+=d;
    g+=.1/(.1+br*13.);
    li+=.1/(.1+tub*5.);
  }
  g=max(g,li*.15);
  float f=1.-td/3.;
  if (ref==1.) f=1.-ltd/3.;
  if (d<.01) {
    col=vec3(1.);
    vec2 e=vec2(0.,det);
    vec3 n=normalize(vec3(de(p+e.yxx),de(p+e.xyx),de(p+e.xxy))-de(p));
    col=vec3(n.x)*.7;
    col+=fract(pos.z*5.)*vec3(.2,.1,.5);
    col+=fractal(pos.xz*2.);
    if (tub>.01) col=vec3(0.);
  }
  col*=f;
  vec3 glo=g*.1*vec3(2.,1.,2.)*(.5+so*1.5)*.5;
  glo.rb*=rot(dir.y*1.5);
  col+=glo;
  col*=vec3(.8,.7,.7);
  col=mix(col,vec3(1.),ref*.3);
  return col;
}

void main( )
{

  vec2 resolution = vec2(0.1, 0.1);
  vec2 uv = vec2(gl_FragCoord.x / resolution.x, gl_FragCoord.y / resolution.y);
  uv -= 0.5;
  uv /= vec2(resolution.y / resolution.x, 1);
  float t=time;
  vec3 from= path(t);
  if (mod(time,10.)>5.) from=path(floor(t/4.+.5)*4.);
  sphpos=path(t+.5);
  from.x+=.2;
  vec3 fw=normalize(path(t+.5)-from);
  vec3 dir=normalize(vec3(uv,.5));
  dir=lookat(fw,vec3(fw.x*2.,1.,0.))*dir;
  dir.xz+=sin(time)*.3;
  vec3 col=march(from,dir);
  col=mix(vec3(.5)*length(col),col,.8);
  fragColor =vec4(col,1.);
}
"""

fragment_shader2 = """

#version 460

layout (location = 0) out vec4 fragColor;
in vec3 ourColor;
in vec2 fragCoord;

uniform float iTime;

float opSmoothUnion( float d1, float d2, float k )
{
    float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h);
}

float sdSphere( vec3 p, float s )
{
  return length(p)-s;
} 

float map(vec3 p)
{
	float d = 2.0;
	for (int i = 0; i < 16; i++) {
		float fi = float(i);
		float time = iTime * (fract(fi * 412.531 + 0.513) - 0.5) * 2.0;
		d = opSmoothUnion(
            sdSphere(p + sin(time + fi * vec3(52.5126, 64.62744, 632.25)) * vec3(2.0, 2.0, 0.8), mix(0.5, 1.0, fract(fi * 412.531 + 0.5124))),
			d,
			0.4
		);
	}
	return d;
}

vec3 calcNormal( in vec3 p )
{
    const float h = 1e-5; // or some other value
    const vec2 k = vec2(1,-1);
    return normalize( k.xyy*map( p + k.xyy*h ) + 
                      k.yyx*map( p + k.yyx*h ) + 
                      k.yxy*map( p + k.yxy*h ) + 
                      k.xxx*map( p + k.xxx*h ) );
}

void main( )
{
    vec2 iResolution = vec2(0.5, 0.5);
    vec2 uv = fragCoord/iResolution.xy;
    
    // screen size is 6m x 6m
	vec3 rayOri = vec3((uv - 0.5) * vec2(iResolution.x/iResolution.y, 1.0) * 6.0, 3.0);
	vec3 rayDir = vec3(0.0, 0.0, -1.0);
	
	float depth = 0.0;
	vec3 p;
	
	for(int i = 0; i < 64; i++) {
		p = rayOri + rayDir * depth;
		float dist = map(p);
        depth += dist;
		if (dist < 1e-6) {
			break;
		}
	}
	
    depth = min(6.0, depth);
	vec3 n = calcNormal(p);
    float b = max(0.0, dot(n, vec3(0.577)));
    vec3 col = (0.5 + 0.5 * cos((b + iTime * 3.0) + uv.xyx * 2.0 + vec3(0,2,4))) * (0.85 + b * 0.35);
    col *= exp( -depth * 0.15 );
	
    // maximum thickness is 2m in alpha channel
    fragColor = vec4(col, 1.0 - (depth - 0.5) / 2.0);
}
"""

fragment_shader3 = """

#version 460

layout (location = 0) out vec4 fragColor;
in vec3 ourColor;
in vec2 fragCoord;

uniform float iTime;
precision highp float;


float gTime = 0.;
const float REPEAT = 5.0;

// 回転行列
mat2 rot(float a) {
	float c = cos(a), s = sin(a);
	return mat2(c,s,-s,c);
}

float sdBox( vec3 p, vec3 b )
{
	vec3 q = abs(p) - b;
	return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float box(vec3 pos, float scale) {
	pos *= scale;
	float base = sdBox(pos, vec3(.4,.4,.1)) /1.5;
	pos.xy *= 5.;
	pos.y -= 3.5;
	pos.xy *= rot(.75);
	float result = -base;
	return result;
}

float box_set(vec3 pos, float iTime) {
	vec3 pos_origin = pos;
	pos = pos_origin;
	pos .y += sin(gTime * 0.4) * 2.5;
	pos.xy *=   rot(.8);
	float box1 = box(pos,2. - abs(sin(gTime * 0.4)) * 1.5);
	pos = pos_origin;
	pos .y -=sin(gTime * 0.4) * 2.5;
	pos.xy *=   rot(.8);
	float box2 = box(pos,2. - abs(sin(gTime * 0.4)) * 1.5);
	pos = pos_origin;
	pos .x +=sin(gTime * 0.4) * 2.5;
	pos.xy *=   rot(.8);
	float box3 = box(pos,2. - abs(sin(gTime * 0.4)) * 1.5);	
	pos = pos_origin;
	pos .x -=sin(gTime * 0.4) * 2.5;
	pos.xy *=   rot(.8);
	float box4 = box(pos,2. - abs(sin(gTime * 0.4)) * 1.5);	
	pos = pos_origin;
	pos.xy *=   rot(.8);
	float box5 = box(pos,.5) * 6.;	
	pos = pos_origin;
	float box6 = box(pos,.5) * 6.;	
	float result = max(max(max(max(max(box1,box2),box3),box4),box5),box6);
	return result;
}

float map(vec3 pos, float iTime) {
	vec3 pos_origin = pos;
	float box_set1 = box_set(pos, iTime);

	return box_set1;
}


void main() {
    vec2 iResolution = vec2(1, 1);
	vec2 p = (fragCoord.xy * 2. - iResolution.xy) / min(iResolution.x, iResolution.y);
	vec3 ro = vec3(0., -0.2 ,iTime * 4.);
	vec3 ray = normalize(vec3(p, 1.5));
	ray.xy = ray.xy * rot(sin(iTime * .03) * 5.);
	ray.yz = ray.yz * rot(sin(iTime * .05) * .2);
	float t = 0.1;
	vec3 col = vec3(0.);
	float ac = 0.0;


	for (int i = 0; i < 99; i++){
		vec3 pos = ro + ray * t;
		pos = mod(pos-2., 4.) -2.;
		gTime = iTime -float(i) * 0.01;
		
		float d = map(pos, iTime);

		d = max(abs(d), 0.01);
		ac += exp(-d*23.);

		t += d* 0.55;
	}

	col = vec3(ac * 0.02);

	col +=vec3(0.,0.2 * abs(sin(iTime)),0.5 + sin(iTime) * 0.2);


	fragColor = vec4(col ,1.0 - t * (0.02 + 0.02 * sin (iTime)));
}
"""

pharaoh = Obj('./Pharaoh.obj')
compiled_vertex_shader = compileShader(vertex_shader, GL_VERTEX_SHADER)
compiled_fragment_shader1 = compileShader(fragment_shader1, GL_FRAGMENT_SHADER)
compiled_fragment_shader2 = compileShader(fragment_shader2, GL_FRAGMENT_SHADER)
compiled_fragment_shader3 = compileShader(fragment_shader3, GL_FRAGMENT_SHADER)


shader1 = compileProgram(
    compiled_vertex_shader,
    compiled_fragment_shader1
)

shader2 = compileProgram(
    compiled_vertex_shader,
    compiled_fragment_shader2
)

shader3 = compileProgram(
    compiled_vertex_shader,
    compiled_fragment_shader3
)

glUseProgram(shader1)

vertex = [element for ver in pharaoh.vertices for element in ver]
vertex_data = numpy.array(vertex, dtype=numpy.float32)

vertex_buffer_object = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object)
glBufferData(
    GL_ARRAY_BUFFER,  # tipo de datos
    vertex_data.nbytes,  # tamaño de da data en bytes
    vertex_data,  # puntero a la data
    GL_STATIC_DRAW
)
vertex_array_object = glGenVertexArrays(1)
glBindVertexArray(vertex_array_object)

glVertexAttribPointer(
    0,
    3,
    GL_FLOAT,
    GL_FALSE,
    3 * 4,
    ctypes.c_void_p(0)
)
glEnableVertexAttribArray(0)


glVertexAttribPointer(
    1,
    3,
    GL_FLOAT,
    GL_FALSE,
    3 * 4,
    ctypes.c_void_p(3 * 4)
)
glEnableVertexAttribArray(1)

faces = [int(f[0]-1) for face in pharaoh.faces for f in face]
faces_data = numpy.array(faces, dtype=numpy.int32)

element_buffer_object = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer_object)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces_data.nbytes,
             faces_data, GL_STATIC_DRAW)


def calculateMatrix(angle, rotation, translation):
    i = glm.mat4(1)
    translate = glm.translate(i, glm.vec3(translation))
    rotate = glm.rotate(i, glm.radians(angle), glm.vec3(rotation))
    scale = glm.scale(i, glm.vec3(0.2, 0.2, 0.2))

    model = translate * rotate * scale

    view = glm.lookAt(
        glm.vec3(0, 0, 5),
        glm.vec3(0, 0, 0),
        glm.vec3(0, 1, 0)
    )

    projection = glm.perspective(
        glm.radians(45),
        700/500,
        0.1,
        1000.0
    )

    amatrix = projection * view * model

    glUniformMatrix4fv(
        glGetUniformLocation(shader1, 'amatrix'),
        1,
        GL_FALSE,
        glm.value_ptr(amatrix)
    )


# glViewport(-165, -250, 1000, 1000)
glViewport(-45, -50, 750, 550)


running = True

glClearColor(0, 1, 0, 1.0)
r = 0
activeShader = shader1
shaderKey = 0
shaderDict = {1: shader1, 2: shader2, 3: shader3}
nextShader = False
prev_iTime = pygame.time.get_ticks()
rotation = (0, 1, 0)
translation = (0, 0, 0)
translation1 = True
translation2 = False
translation3 = False

while running:
    r += 1
    glClear(GL_COLOR_BUFFER_BIT)

    if nextShader:
        shaderOn = shaderDict.get(shaderKey)
        glUseProgram(shaderOn)
        activeShader = shaderOn
        nextShader = False

    color1 = random.random()
    color2 = random.random()
    color3 = random.random()

    color = glm.vec3(color1, color2, color3)

    glUniform3fv(
        glGetUniformLocation(activeShader, 'color'),
        1,
        glm.value_ptr(color)
    )

    iTime = (pygame.time.get_ticks() - prev_iTime) / 1000
    glUniform1f(
        glGetUniformLocation(activeShader, "iTime"),
        iTime
    )

    if translation1:
        translation = (0, sin(r/15), 0)
        rotation = (1, 0, 0)
        calculateMatrix(6*r, rotation, translation)
    if translation2:
        translation = (sin(r/15), 0, 0)
        rotation = (0, 0, 1)
        calculateMatrix(r, rotation, translation)
    if translation3:
        translation = (-sin(r/15), cos(r/20), -cos(r/20))
        rotation = (1, 0, 1)
        calculateMatrix(-r, rotation, translation)

    pygame.time.wait(50)

    glDrawElements(GL_TRIANGLES, len(faces_data), GL_UNSIGNED_INT, None)

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                shaderKey = 1
                nextShader = True
                translation1 = True
                translation2 = False
                translation3 = False
            if event.key == pygame.K_2:
                shaderKey = 2
                nextShader = True
                translation1 = False
                translation2 = True
                translation3 = False
            if event.key == pygame.K_3:
                shaderKey = 3
                nextShader = True
                translation1 = False
                translation2 = False
                translation3 = True

        if event.type == pygame.QUIT:
            running = False
