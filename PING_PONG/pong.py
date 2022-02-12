# Simple pong Games

import turtle
import os
# for windows  - Import winsound

wn = turtle.Screen()
wn.title("Pong by Ashu")
wn.bgcolor("black")
wn.setup(width=800,height=600)
wn.tracer(0)

#Score
score_a=0
score_b=0

# Paddle A 
paddle_a = turtle.Turtle()
paddle_a.speed(0)   # speed of animations
paddle_a.shape("square")
paddle_a.color("white")
paddle_a.shapesize(stretch_wid=5,stretch_len=1)
paddle_a.penup()   # stop to draw a line while moving
paddle_a.goto(-350,0)
# Paddle B
paddle_b = turtle.Turtle()
paddle_b.speed(0)   # speed of animations
paddle_b.shape("square")
paddle_b.color("white")
paddle_b.shapesize(stretch_wid=5,stretch_len=1)
paddle_b.penup()   # stop to draw a line while moving
paddle_b.goto(350,0)
# Ball
ball = turtle.Turtle()
ball.speed(0)   # speed of animations
ball.shape("square")
ball.color("white")
ball.penup()   # stop to draw a line while moving
ball.goto(0,0)
ball.dx = 0.2  #every time ball moves
ball.dy = -0.2

# Pen
pen = turtle.Turtle()
pen.speed(0)
pen.color("white")
pen.penup()
pen.hideturtle()
pen.goto(0,260)
pen.write("Player A: 0  Player B: 0" , align="center", font=("Courier",24,"normal"))

# Function 
#Moving the Paddle with Keys

def paddle_a_up():
    y = paddle_a.ycor()    # from turtle module  y cordinate
    y += 20
    paddle_a.sety(y)
    
def paddle_a_down():
    y = paddle_a.ycor()    # from turtle module  y cordinate
    y -= 20
    paddle_a.sety(y)
    
def paddle_b_up():
    y = paddle_b.ycor()    # from turtle module  y cordinate
    y += 20
    paddle_b.sety(y)
    
def paddle_b_down():
    y = paddle_b.ycor()    # from turtle module  y cordinate
    y -= 20
    paddle_b.sety(y)
    


    
    
# Keyboard binding
wn.listen()
wn.onkeypress(paddle_a_up,"w")
wn.onkeypress(paddle_a_down,"s")
wn.onkeypress(paddle_b_up,"Up")
wn.onkeypress(paddle_b_down,"Down")



#Main Games loop
while True:
    wn.update()
    
    # Move the ball
    ball.setx(ball.xcor()+ball.dx)
    ball.sety(ball.ycor() + ball.dy )
    
    # Border Checking
    if ball.ycor() > 290:   # Top border
        ball.sety(290)
        ball.dy *=-1    # revrse the direction of ball
        os.system("afplay bounce.wav&")   # aplay for linux for windowds - winsound.PlaySound("file_name",winsound.SND_ASYNC)
        
    if ball.ycor() < -290:   # bottom border
        ball.sety(-290)
        ball.dy *=-1    # revrse the direction of ball
        os.system("afplay bounce.wav&")
    
    if ball.xcor() > 390:
        ball.goto(0,0)
        ball.dx *=-1
        score_a +=1
        pen.clear()
        pen.write("Player A: {}  Player B: {}".format(score_a,score_b) , align="center", font=("Courier",24,"normal"))
    
    if ball.xcor() < -390:
        ball.goto(0,0)
        ball.dx *=-1
        score_b +=1
        pen.clear()
        pen.write("Player A: {}  Player B: {}".format(score_a,score_b) , align="center", font=("Courier",24,"normal"))
        
    # paddle and ball colliisions
    # paddle B
    if ball.xcor() >340 and ball.xcor() < 350 and (ball.ycor() < paddle_b.ycor()+40 and ball.ycor() > paddle_b.ycor()-40):
        ball.setx(340)
        ball.dx *= -1
        os.system("afplay bounce.wav&")
        
    #paddle A
    if ball.xcor() < -340 and ball.xcor() > -350 and (ball.ycor() < paddle_a.ycor()+40 and ball.ycor() > paddle_b.ycor()-40):
        ball.setx(-340)
        ball.dx *= -1
        os.system("afplay bounce.wav&")