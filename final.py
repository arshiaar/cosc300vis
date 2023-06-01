import glfw
import glfw.GLFW as GLFW_CONSTANTS
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
from PIL import Image
import random

import re


def load_model_from_file(
        filename: str):
    """
        Read the given obj file and return a list of all the
        vertex data.
    """

    v = []
    vt = []
    vn = []
    vertices = []

    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            words = line.split(" ")
            if words[0] == "v":
                v.append(read_vertex_data(words))
            elif words[0] == "vt":
                vt.append(read_texcoord_data(words))
            elif words[0] == "vn":
                vn.append(read_normal_data(words))
            elif words[0] == "f":
                read_face_data(words, v, vt, vn, vertices)
            line = f.readline()

    return vertices


def read_vertex_data(words):
    """
        read the given position description and
        return the vertex it represents.
    """

    return [
        float(words[1]),
        float(words[2]),
        float(words[3])
    ]


def read_texcoord_data(words):
    """
        read the given texcoord description and
        return the texcoord it represents.
    """

    return [
        float(words[1]),
        float(words[2])
    ]


def read_normal_data(words):
    """
        read the given normal description and
        return the normal it represents.
    """

    return [
        float(words[1]),
        float(words[2]),
        float(words[3])
    ]


def read_face_data(
        words,
        v, vt, vn,
        vertices) -> None:
    """
        Read the given face description, and use the
        data from the pre-filled v, vt, vn arrays to add
        data to the vertices array
    """

    triangles_in_face = len(words) - 3

    for i in range(triangles_in_face):
        read_corner(words[1], v, vt, vn, vertices)
        read_corner(words[i + 2], v, vt, vn, vertices)
        read_corner(words[i + 3], v, vt, vn, vertices)


def read_corner(
        description: str,
        v, vt, vn,
        vertices) -> None:
    """
        Read the given corner description, then send the
        approprate v, vt, vn data to the vertices array.
    """

    v_vt_vn = description.split("/")

    for x in v[int(v_vt_vn[0]) - 1]:
        vertices.append(x)
    for x in vt[int(v_vt_vn[1]) - 1]:
        vertices.append(x)
    for x in vn[int(v_vt_vn[2]) - 1]:
        vertices.append(x)

def createShader(vertexFilepath: str, fragmentFilepath: str) -> int:
    """
        Compile and link a shader program from source.

        Parameters:

            vertexFilepath: filepath to the vertex shader source code (relative to this file)

            fragmentFilepath: filepath to the fragment shader source code (relative to this file)

        Returns:

            An integer, being a handle to the shader location on the graphics card
    """

    with open(vertexFilepath, 'r') as f:
        vertex_src = f.readlines()

    with open(fragmentFilepath, 'r') as f:
        fragment_src = f.readlines()

    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                            compileShader(fragment_src, GL_FRAGMENT_SHADER))

    return shader


class Entity:
    """ Represents a general object with a position and rotation applied"""

    def __init__(self, position, eulers):
        """
            Initialize the entity, store its state and update its transform.

            Parameters:

                position: The position of the entity in the world (x,y,z)

                eulers: Angles (in degrees) representing rotations around the x,y,z axes.
        """

        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)

    def get_model_transform(self) -> np.ndarray:
        """
            Calculates and returns the entity's transform matrix,
            based on its position and rotation.
        """

        model_transform = pyrr.matrix44.create_identity(dtype=np.float32)

        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_z_rotation(
                theta=np.radians(self.eulers[2]),
                dtype=np.float32
            )
        )

        model_transform = pyrr.matrix44.multiply(
            m1=model_transform,
            m2=pyrr.matrix44.create_from_translation(
                vec=self.position,
                dtype=np.float32
            )
        )

        return model_transform

    def update(self, rate: float) -> None:
        raise NotImplementedError


class GroundTile(Entity):

    def __init__(
            self, position,
            eulers, scale):
        """ Initialize a triangle with the given scale."""

        super().__init__(position, eulers)
        self.scale = np.array(scale, dtype=np.float32)

    def update(self, rate: float) -> None:
        """
            Update the triangle.

            Parameters:

                rate: framerate correction factor
        """

        pass

    def get_model_transform(self) -> np.ndarray:

        return pyrr.matrix44.multiply(
            m1=pyrr.matrix44.create_from_scale(
                scale=self.scale,
                dtype=np.float32
            ),
            m2=super().get_model_transform()
        )


class Triangle(Entity):
    """ A triangle that spins. """

    def __init__(
            self, position,
            eulers, scale):
        """ Initialize a triangle with the given scale."""

        super().__init__(position, eulers)
        self.scale = np.array(scale, dtype=np.float32)

    def update(self, rate: float) -> None:
        """
            Update the triangle.

            Parameters:

                rate: framerate correction factor
        """

        pass

    def get_model_transform(self) -> np.ndarray:
        return pyrr.matrix44.multiply(
            m1=pyrr.matrix44.create_from_scale(
                scale=self.scale,
                dtype=np.float32
            ),
            m2=super().get_model_transform()
        )

class Light:

    def __init__(self, position, color, strength):
        self.position = np.array(position, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32)
        self.strength = strength


class Player(Triangle):
    """ A player character """

    def __init__(self, position, eulers, scale):
        """ Initialise a player character. """

        super().__init__(position, eulers, scale)
        self.camera = None

    def update(self, target: Triangle, rate: float) -> None:
        """
            Update the player.

            Parameters:

                target: the triangle to move towards.

                rate: framerate correction factor
        """

        if target is not None:
            print(target.position)
            self.move_towards(target.position, 0.1 * rate)

    def move_towards(self, targetPosition: np.ndarray, amount: float) -> None:
        """
            Move towards the given point by the given amount.
        """
        directionVector = targetPosition - self.position
        angle = np.arctan2(-directionVector[1], directionVector[0])
        self.move(angle, amount)

    def move(self, direction: float, amount: float) -> None:
        """
            Move by the given amount in the given direction (in radians).
        """
        self.position[0] += amount * np.cos(direction, dtype=np.float32)
        self.position[1] -= amount * np.sin(direction, dtype=np.float32)
        self.camera.position[0] += amount * np.cos(direction, dtype=np.float32)
        self.camera.position[1] -= amount * np.sin(direction, dtype=np.float32)
        self.eulers[2] = np.degrees(direction) - 45


class Camera(Entity):
    """ A third person camera controller. """

    def __init__(self, position):
        super().__init__(position, eulers=[0, 0, 0])

        self.forwards = np.array([0, 0, 0], dtype=np.float32)
        self.right = np.array([0, 0, 0], dtype=np.float32)
        self.up = np.array([0, 0, 0], dtype=np.float32)

        self.localUp = np.array([0, 0, 1], dtype=np.float32)
        self.targetObject: Entity = None

    def update(self) -> None:
        """ Updates the camera """

        self.calculate_vectors_cross_product()

    def calculate_vectors_cross_product(self) -> None:
        """
            Calculate the camera's fundamental vectors.

            There are various ways to do this, this function
            achieves it by using cross products to produce
            an orthonormal basis.
        """

        self.forwards = pyrr.vector.normalize(self.targetObject.position - self.position)
        self.right = pyrr.vector.normalize(pyrr.vector3.cross(self.forwards, self.localUp))
        self.up = pyrr.vector.normalize(pyrr.vector3.cross(self.right, self.forwards))

    def get_view_transform(self) -> np.ndarray:
        """ Return's the camera's view transform. """

        return pyrr.matrix44.create_look_at(
            eye=self.position,
            target=self.targetObject.position,
            up=self.up,
            dtype=np.float32
        )


class Scene:
    """
        Manages all logical objects in the game,
        and their interactions.
    """

    def __init__(self):
         # when rendering - render with the wheel mesh.

        self.player = Player(
            position=[0, 1, 0],
            eulers=[0, 0, 0],
            scale=[0.5, 0.5, 0.5]
        )
        self.camera = Camera(position=[-3, 1, 3])
        self.player.camera = self.camera
        self.camera.targetObject = self.player

        # create triangle mesh
        self.tile = GroundTile(position=[0, 1, 0],
                               eulers=[0, 0, 0],
                               scale=[1, 1, 1])

        self.click_dots: list[Entity] = []

        # make row of triangles
        # here - create the multiple item boxes here.

        self.triangles: list[Entity] = []
        self.triangles.append(
            Triangle(
                position=[10, 1.5, 0.05],
                eulers=[0, 0, 0],
                scale=[0.5, 0.5, 0.55],
            )
        )
        self.triangles.append(
            Triangle(
                position=[10, 0.75, 0.05],
                eulers=[0, 0, 0],
                scale=[0.5, 0.5, 0.55],
            )
        )
        self.triangles.append(
            Triangle(
                position=[10, 0, 0.05],
                eulers=[0, 0, 0],
                scale=[0.5, 0.5, 0.5],
            )
        )
        self.triangles.append(
            Triangle(
                position=[10, -0.75, 0.05],
                eulers=[0, 0, 0],
                scale=[0.5, 0.5, 0.5],
            )
        )

        # and now also horizontal traingles.
        self.triangles.append(
            Triangle(
                position=[15, -8, 0.05],
                eulers=[0, 0, 0],
                scale=[0.5, 0.5, 0.55],
            )
        )
        self.triangles.append(
            Triangle(
                position=[14, -8, 0.05],
                eulers=[0, 0, 0],
                scale=[0.5, 0.5, 0.55],
            )
        )
        self.triangles.append(
            Triangle(
                position=[13, -8, 0.05],
                eulers=[0, 0, 0],
                scale=[0.5, 0.5, 0.5],
            )
        )
        self.triangles.append(
            Triangle(
                position=[12, -8, 0.05],
                eulers=[0, 0, 0],
                scale=[0.5, 0.5, 0.5],
            )
        )

        self.shells = []
        self.shells.append(Triangle(
                 position=[5, 0, 0.05],
                 eulers=[0, 0, 0],
                 scale=[0.2, 0.2, 0.2],
             ))
        self.shells.append(Triangle(
             position=[13, -7, 0.05],
             eulers=[0, 0, 0],
             scale=[0.2, 0.2, 0.2],
         ))
        self.shells.append(Triangle(
             position=[4, -9, 0.05],
             eulers=[0, 0, 0],
             scale=[0.2, 0.2, 0.2],
         ))



        self.lights = [
            Light(
                position=[4, 0, 5],
                color=[1.0, 0.8, 0.5],
                strength=15
            ),
        ]




    def update(self, rate: float) -> None:
        """
            Update all objects managed by the scene.

            Parameters:

                rate: framerate correction factor
        """

        for triangle in self.triangles:
            triangle.update(rate)
        for dot in self.click_dots:
            dot.update(rate)

        # also update the ground tile here.
        self.tile.update(rate)

        targetDot = None
        if len(self.click_dots) > 0:
            targetDot = self.click_dots[0]
        self.player.update(targetDot, rate)
        self.camera.update()

        # check if dot can be deleted
        if targetDot is not None:
            if pyrr.vector.length(targetDot.position - self.player.position) < 0.1:
                self.click_dots.pop(self.click_dots.index(targetDot))

    def lay_down_dot(self, position) -> None:
        """ Drop a dot at the given position """

        self.click_dots.append(
            Triangle(
                position=position,
                eulers=[0, 0, 0],
                scale=[0.00001, 0.00001, 0.00001],
            )
        )

    def move_camera(self, dPos: np.ndarray) -> None:
        """
            Move the camera by the given amount in its fundamental vectors.
        """

        self.camera.position += dPos[0] * self.camera.forwards \
                                + dPos[1] * self.camera.right \
                                + dPos[2] * self.camera.up


class App:
    """ The main program """

    def __init__(self):
        """ Set up the program """

        self.set_up_glfw()

        self.make_assets()

        self.set_onetime_uniforms()

        self.get_uniform_locations()

        self.set_up_input_systems()

        self.set_up_timer()

        self.mainLoop()

    def set_up_glfw(self) -> None:
        """ Set up the glfw environment """

        self.screenWidth = 640
        self.screenHeight = 480

        glfw.init()
        glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(
            GLFW_CONSTANTS.GLFW_OPENGL_PROFILE,
            GLFW_CONSTANTS.GLFW_OPENGL_CORE_PROFILE
        )
        glfw.window_hint(
            GLFW_CONSTANTS.GLFW_OPENGL_FORWARD_COMPAT,
            GLFW_CONSTANTS.GLFW_TRUE
        )
        glfw.window_hint(GLFW_CONSTANTS.GLFW_DOUBLEBUFFER, False)
        self.window = glfw.create_window(
            self.screenWidth, self.screenHeight, "Title", None, None
        )
        glfw.make_context_current(self.window)

    def make_assets(self) -> None:
        """ Make any assets used by the App"""

        self.scene = Scene()
        self.triangle_mesh = TriangleMesh()
        self.triangle_fan_mesh = TriangleFanMesh()
        self.wall_mesh = WallMesh()

        # all other meshes.
        self.pond_mesh = PondMesh()
        self.pond_edge_mesh = PondEdgeMesh()
        self.track_mesh = TrackMesh()
        self.sky_mesh = Quad2D(center = (0,0),
                size = (1,1))




        # render
        self.item_mesh = ItemMesh()
        self.shell_mesh = ShellMesh()
        self.player_mesh = PlayerMesh()

        # create a sky shader and then go from there.
        # and then use this exact information to create the shader.
        # create a sky shader - and then go from there.
        self.shader = createShader("shaders/vertex.txt", "shaders/fragment.txt")


    def set_onetime_uniforms(self) -> None:
        """ Set any uniforms which can simply get set once and forgotten """

        glUseProgram(self.shader)

        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy=45,
            aspect=self.screenWidth / self.screenHeight,
            near=0.1, far=50, dtype=np.float32
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "projection"),
            1, GL_FALSE, projection_transform
        )
        # okay - so just setting these here.

        self.lightLocation = {
            "position": glGetUniformLocation(self.shader, "Light.position"),
            "color": glGetUniformLocation(self.shader, "Light.color"),
            "strength": glGetUniformLocation(self.shader, "Light.strength")
        }
        self.cameraPosLoc = glGetUniformLocation(self.shader, "cameraPostion")




    def get_uniform_locations(self) -> None:
        """ Query and store the locations of any uniforms on the shader """

        glUseProgram(self.shader)
        self.modelMatrixLocation = glGetUniformLocation(self.shader, "model")
        self.viewMatrixLocation = glGetUniformLocation(self.shader, "view")

        # okay - so just store the rest of the stuff here.




    def set_up_input_systems(self) -> None:

        glfw.set_mouse_button_callback(self.window, self.handleMouse)

    def set_up_timer(self) -> None:
        """
            Set up the variables needed to measure the framerate
        """
        self.lastTime = glfw.get_time()
        self.currentTime = 0
        self.numFrames = 0
        self.frameTime = 0

    def mainLoop(self):

        glClearColor(0.1, 0.2, 0.2, 1)
        (w, h) = glfw.get_framebuffer_size(self.window)
        glViewport(0, 0, w, h)
        glEnable(GL_DEPTH_TEST)
        running = True



        while (running):

            # check events
            if glfw.window_should_close(self.window) \
                    or glfw.get_key(
                self.window, GLFW_CONSTANTS.GLFW_KEY_ESCAPE
            ) == GLFW_CONSTANTS.GLFW_PRESS:
                running = False

            self.handleKeys()

            glfw.poll_events()

            # update scene
            self.scene.update(self.frameTime / 16.667)

            # refresh screen
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glUseProgram(self.shader)

            # get light, and the position etc required.
            light = self.scene.lights[0]
            glUniform3fv(self.lightLocation["position"], 1, light.position)
            glUniform3fv(self.lightLocation["color"], 1, light.color)
            glUniform1f(self.lightLocation["strength"], light.strength)

            # being passed to the shader.
            glUniformMatrix4fv(
                self.viewMatrixLocation, 1, GL_FALSE,
                self.scene.camera.get_view_transform()
            )

            glUniformMatrix4fv(
                self.modelMatrixLocation,
                1, GL_FALSE,
                self.scene.player.get_model_transform()
            )

            texture_uniform_location = glGetUniformLocation(self.shader, "imageTexture")


            # TRIANGLE FAN START
            glBindVertexArray(self.player_mesh.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.player_mesh.vertex_count)
            glBindVertexArray(0)

            # being passed to the shader.
            glUniformMatrix4fv(
                self.viewMatrixLocation, 1, GL_FALSE,
                self.scene.camera.get_view_transform()
            )

            glUniformMatrix4fv(
                self.modelMatrixLocation,
                1, GL_FALSE,
                self.scene.tile.get_model_transform()
            )
           # glBindVertexArray(self.triangle_fan_mesh.vao)

            # here the drawing changes - bu
            #glDrawArrays(GL_TRIANGLES, 0, self.triangle_fan_mesh.vertex_count)
            #glBindVertexArray(0)
            # TRIANGLE FAN END

            # TODO POND MESH
            glBindVertexArray(self.pond_mesh.vao)
            glBindTexture(GL_TEXTURE_2D, self.pond_mesh.texture)
            glUniform1i(texture_uniform_location, 0)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, self.pond_mesh.vertex_count)
            glBindVertexArray(0)

            # TODO POND MESH END
            glUniformMatrix4fv(
                self.viewMatrixLocation, 1, GL_FALSE,
                self.scene.camera.get_view_transform()
            )

            glUniformMatrix4fv(
                self.modelMatrixLocation,
                1, GL_FALSE,
                self.scene.tile.get_model_transform()
            )

            # TODO - do this for - pond_mesh, track_mesh, pond_edge_mesh, cube_mesh
            glBindVertexArray(self.track_mesh.vao)
            glBindTexture(GL_TEXTURE_2D, self.track_mesh.texture)
            glUniform1i(texture_uniform_location, 0)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, self.track_mesh.vertex_count)
            glBindVertexArray(0)

            # TODO - do this for - pond_mesh, track_mesh, pond_edge_mesh, cube_mesh
            # TRIANGLE FAN START
            glBindVertexArray(self.pond_edge_mesh.vao)
            glBindTexture(GL_TEXTURE_2D, self.pond_edge_mesh.texture)
            glUniform1i(texture_uniform_location, 0)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, self.pond_edge_mesh.vertex_count)
            glBindVertexArray(0)

            # WALL MESH
            glBindVertexArray(self.wall_mesh.vao)
            glBindTexture(GL_TEXTURE_2D, self.wall_mesh.texture)
            glUniform1i(texture_uniform_location, 0)
            # here the drawing changes - bu
            glDrawArrays(GL_TRIANGLE_STRIP, 0, self.wall_mesh.vertex_count)
            glBindVertexArray(0)

            # query the texture location
            # and then update.
            sky_uniform_location = glGetUniformLocation(self.shader, "imageTexture")
            # and now actually render the sky
            # draw sky
            glBindVertexArray(self.sky_mesh.vao)
            glBindTexture(GL_TEXTURE_2D, self.sky_mesh.texture)
            glUniform1i(sky_uniform_location, 0)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, self.sky_mesh.vertex_count)
            glBindVertexArray(0)

            # WALL MESH
            #glBindVertexArray(self.item_mesh.vao)
            #glBindTexture(GL_TEXTURE_2D, self.item_mesh.texture)
            #glUniform1i(texture_uniform_location, 0)
            #glDrawArrays(GL_TRIANGLE_STRIP, 0, self.item_mesh.vertex_count)
            #glBindVertexArray(0)

            # fix that later
            # triangle mesh -> these are being drawn using the wall texture.
            # this needs to have a texture?
            ### HERE - BEGINS RENDERING WITH COLOUR ONLY.


            # render the item box
            # Bind the vertex array object


            # Disable texture mapping

            # TODO - shell mesh, get it in there
            glBindTexture(GL_TEXTURE_2D, 0)
            # set up material uniforms


            # create a liost of cells and then iterate and do this.
            for shell in self.scene.shells:
                glUniformMatrix4fv(
                    self.modelMatrixLocation,
                     1, GL_FALSE, shell.get_model_transform()
                )
                self.shell_mesh.render(self.shader)

            #glBindVertexArray(self.shell_mesh.vao)
            # here the drawing changes - bu
            #glDrawArrays(GL_TRIANGLES, 0, self.shell_mesh.vertex_count)
            #glBindVertexArray(0)






            # render these as question marks.
            for triangle in self.scene.triangles:
                glUniformMatrix4fv(
                    self.modelMatrixLocation,
                    1, GL_FALSE,
                    triangle.get_model_transform()
                )
                glBindVertexArray(self.item_mesh.vao)
                glDrawArrays(GL_TRIANGLES, 0, self.item_mesh.vertex_count)

            # render these as blue objects.

            for dot in self.scene.click_dots:
                glUniformMatrix4fv(
                    self.modelMatrixLocation,
                    1, GL_FALSE,
                    dot.get_model_transform()
                )
                glDrawArrays(GL_TRIANGLES, 0, self.triangle_mesh.vertex_count)
            glBindVertexArray(0)


            glFlush()

            # timing
            self.calculateFramerate()
        self.quit()

    def handleKeys(self) -> None:
        """ Handle keys. """

        camera_movement = [0, 0, 0]

        if glfw.get_key(
                self.window, GLFW_CONSTANTS.GLFW_KEY_W
        ) == GLFW_CONSTANTS.GLFW_PRESS:
            camera_movement[2] += 1
        elif glfw.get_key(
                self.window, GLFW_CONSTANTS.GLFW_KEY_A
        ) == GLFW_CONSTANTS.GLFW_PRESS:
            camera_movement[1] -= 1
        elif glfw.get_key(
                self.window, GLFW_CONSTANTS.GLFW_KEY_S
        ) == GLFW_CONSTANTS.GLFW_PRESS:
            camera_movement[2] -= 1
        elif glfw.get_key(
                self.window, GLFW_CONSTANTS.GLFW_KEY_D
        ) == GLFW_CONSTANTS.GLFW_PRESS:
            camera_movement[1] += 1

        dPos = 0.1 * self.frameTime / 16.667 * np.array(
            camera_movement,
            dtype=np.float32
        )

        self.scene.move_camera(dPos)

    def handleMouse(self, window, button: int, action: int, mods: int) -> None:

        if button != GLFW_CONSTANTS.GLFW_MOUSE_BUTTON_LEFT \
                or action != GLFW_CONSTANTS.GLFW_PRESS:
            return

        # fetch camera's vectors
        forward = self.scene.camera.forwards
        up = self.scene.camera.up
        right = self.scene.camera.right

        # get mouse's displacement from screen center
        (x, y) = glfw.get_cursor_pos(self.window)
        rightAmount = (x - self.screenWidth // 2) / self.screenWidth
        upAmount = (self.screenHeight // 2 - y) / self.screenWidth

        # get resultant direction (from camera eye, through point on screen)
        resultant = pyrr.vector.normalize(forward + rightAmount * right + upAmount * up)

        # trace from camera's position until we hit the ground
        if (resultant[2] < 0):
            x = self.scene.camera.position[0]
            y = self.scene.camera.position[1]
            z = self.scene.camera.position[2]
            while (z > 0):
                x += resultant[0]
                y += resultant[1]
                z += resultant[2]
            self.scene.lay_down_dot(
                position=[x, y, 0]
            )

    def calculateFramerate(self) -> None:
        """
            Calculate the framerate and frametime
        """

        self.currentTime = glfw.get_time()
        delta = self.currentTime - self.lastTime
        if (delta >= 1):
            framerate = int(self.numFrames / delta)
            glfw.set_window_title(self.window, f"Running at {framerate} fps.")
            self.lastTime = self.currentTime
            self.numFrames = -1
            self.frameTime = float(1000.0 / max(60, framerate))
        self.numFrames += 1

    def quit(self):
        self.triangle_mesh.destroy()
        glDeleteProgram(self.shader)
        glfw.terminate()


class TriangleMesh:

    def __init__(self):
        # x, y, z, r, g, b
        self.vertices = (
            -0.5, -0.5, 0.0, 5.0, 0.0, 0.0,
            0.5, -0.5, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.5, 0.0, 0.0, 0.0, 1.0
        )
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vertex_count = 3

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))


# triangle plane class
class TriangleFanMesh():

    def __init__(self):

        # triangle fan is actually cube mesh niw!

        # x, y, z, r, g, b
        self.vertices = (


            # Front face
            -0.5, -0.5, 0.5, 0.0, 0.0, 1.0,  # Bottom-left
            0.5, -0.5, 0.5, 1.0, 0.0, 1.0,  # Bottom-right
            -0.5, 0.5, 0.5, 0.0, 1.0, 1.0,  # Top-left
            0.5, 0.5, 0.5, 1.0, 1.0, 1.0,  # Top-right

            # Back face
            -0.5, -0.5, -0.5, 1.0, 0.0, 0.0,  # Bottom-left
            0.5, -0.5, -0.5, 0.0, 1.0, 0.0,  # Bottom-right
            -0.5, 0.5, -0.5, 1.0, 1.0, 0.0,  # Top-left
            0.5, 0.5, -0.5, 0.0, 0.0, 1.0,  # Top-right

            # Left face
            -0.5, -0.5, -0.5, 1.0, 0.0, 0.0,  # Bottom-front
            -0.5, 0.5, -0.5, 0.0, 1.0, 0.0,  # Top-front
            -0.5, -0.5, 0.5, 0.0, 0.0, 1.0,  # Bottom-back
            -0.5, 0.5, 0.5, 1.0, 1.0, 0.0,  # Top-back

            # Right face
            0.5, -0.5, -0.5, 1.0, 1.0, 0.0,  # Bottom-front
            0.5, 0.5, -0.5, 0.0, 0.0, 1.0,  # Top-front
            0.5, -0.5, 0.5, 0.0, 1.0, 0.0,  # Bottom-back
            0.5, 0.5, 0.5, 1.0, 0.0, 0.0,  # Top-back

            # Top face
            -0.5, 0.5, -0.5, 1.0, 1.0, 0.0,  # Bottom-front
            0.5, 0.5, -0.5, 0.0, 0.0, 1.0,  # Top-front
            -0.5, 0.5, 0.5, 0.0, 1.0, 0.0,  # Bottom-back
            0.5, 0.5, 0.5, 1.0, 0.0, 0.0,  # Top-back

            # Bottom face
            -0.5, -0.5, -0.5, 0.0, 0.0, 1.0,  # Bottom-front
            0.5, -0.5, -0.5, 1.0, 1.0, 0.0,  # Top-front
            -0.5, -0.5, 0.5, 1.0, 0.0, 0.0,  # Bottom-back
            0.5, -0.5, 0.5, 0.0, 1.0, 0.0,  # Top-back
        )
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vertex_count = 24

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))

        # this was the mesh code given by chat gpt.

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))


# triangle plane class
class PondMesh():

    def __init__(self):

        # 0.325, 0.396,0.525

        self.vertices = (
            # Outer rectangle
            5, -2.0, -1.5, 0, 0, 0.325, 0.396,0.625, # v1
            10, -2.0, -1.5, 1, 0, 0.325, 0.396,0.625,# v2
            5, -7, -1.5, 0, 1, 0.325, 0.396,0.625 ,# v3
            10, -7, -1.5, 0, 1, 0.325, 0.396,0.625, # v4
            10, -7, -1.5, 0, 1, 0.325, 0.396,0.625, # v4
            14, -9, -1.5, 0, 0,0.325, 0.396,0.625, #,v5
            5, -7, -1.5, 0, 1, 0.325, 0.396,0.625, # v3
            3, -9, -1.5, 1, 0, 0.325, 0.396,0.625, # v6

        )

        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vertex_count = 8

        texture_path = 'yo_wave_tex.png'
        self.texture = self.load_texture(texture_path)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))

    def load_texture(self, path):
        image = Image.open(path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = np.array(list(image.getdata()), np.uint8)

        # Change alpha value
        alpha = 0.1  # Example: Set alpha to 0.4 for 40% transparency
        alpha = np.clip(alpha, 0.0, 1.0)  # Ensure alpha is within range [0.0, 1.0]
        img_data[:, 3] = (alpha * 255).astype(np.uint8)  # Normalize and convert to uint8

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        return texture

    def destroy(self):

        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))
        glDeleteTextures(1, (self.texture,))

class PondEdgeMesh():

    def __init__(self):

        # x, y, z, u, v, r, g, b
        self.vertices = (

            # redo pond edge walls according to these.

            # v1 to v2 wall together.
            5, -2.0, -1.5, 0, 1, 1.0, 0.0, 0.0,  # v1
            5, -2.0, -0.75, 0, 0, 1.0, 0.0, 0.0,  # v1
            10, -2.0, -1.5, 1, 0, 1.0, 0.0, 0.0,  # v2
            10, -2.0, -0.75, 1, 1, 1.0, 0.0, 0.0,  # v2

            # okay - first edge.
            # ends at the top edge of v2.
            10, -7, -1.5, 0, 1, 1.0, 0.0, 0.0,  # v4
            10, -7, -0.75, 0, 0, 1.0, 0.0, 0.0,  # v4


            # okay - so that' also fine.
            # top and end bit of v5 here.
            # v5 up here.
            14, -9, -1.5, 1, 0, 1.0, 0.0, 0.0,  # v5
            14, -9, -0.75, 1, 1, 1.0, 0.0, 0.0,  # v5


            # v5, and the end bits of v6 here.
            3, -9, -1.5, 0, 0, 1.0, 0.0, 0.0,  # v6
            3, -9, -0.75, 0, 1, 1.0, 0.0, 0.0,  # v6


            # v6 connects to v3.
            5, -7, -1.5, 1, 0, 1.0, 0.0, 0.0,  # v3
            5, -7, -0.75, 1, 1, 1.0, 0.0, 0.0,  # v3

            # connect to v1.
            5, -2.0, -1.5, 0, 1, 1.0, 0.0, 0.0,  # v1
            5, -2.0, -0.75, 0, 0, 1.0, 0.0, 0.0,  # v1


        )
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vertex_count = 14

        texture_path = 's64_iwa.png'
        self.texture = self.load_texture(texture_path)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))

    def load_texture(self, path):
        image = Image.open(path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = np.array(list(image.getdata()), np.uint8)

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        return texture

    def destroy(self):

        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))
        glDeleteTextures(1, (self.texture,))

# triangle plane class
class TrackMesh():

    def __init__(self):

        # x, y, z, u, v, r, g, b
        self.vertices = (
            # lets figure out the track mesh




            # A1.3
            -4.0, 2.0, -0.75, 0, 0, 1.0, 0.0, 0.0,
            -4.0, -2.0, -0.75, 0, 0, 1.0, 0.0, 0.0,
            5.0, 2.0, -0.75, 1, 0, 1.0, 0.0, 0.0,
            5.0, -2.0, -0.75, 1, 1, 1.0, 0.0, 0.0,


            # A1 - just the two additional vertices.
            10, 2.0, -0.75, 0, 1, 1.0, 0.0, 0.0,
            10, -2.0, -0.75, 0, 1, 1.0, 0.0, 0.0,


            # A1.2
            16, 2.0, -0.75, 0, 1, 1.0, 0.0, 0.0,
            16, -2.0, -0.75, 0, 1, 1.0, 0.0, 0.0,

            10, -2.0, -0.75, 0, 1, 1.0, 0.0, 0.0, # specify again just so we have that vertex there.

            # two additional ones.
            16, -7.0, -0.75, 0, 1, 1.0, 0.0, 0.0,
            10, -7.0, -0.75, 0, 1, 1.0, 0.0, 0.0,
            10, -7.0, -0.75, 0, 1, 1.0, 0.0, 0.0,  # copy to avoid compliactions


            # then A8
            10, -7.0, -0.75, 0, 1, 1.0, 0.0, 0.0,
            14, -9.0, -0.75, 0, 1, 1.0, 0.0, 0.0,
            14, -7.0, -0.75, 0, 1, 1.0, 0.0, 0.0,


            # leads into a5 nicely.
            16, -9.0, -0.75, 0, 1, 1.0, 0.0, 0.0,
            16, -7.0, -0.75, 0, 1, 1.0, 0.0, 0.0,

            16, -9.0, -0.75, 0, 1, 1.0, 0.0, 0.0, # duplicate to avoid complications

            # get a6 done here.
            # mere bache, everything will be okay.
            16, -12.0, -0.75, 0, 1, 1.0, 0.0, 0.0,
            16, -7.0, -0.75, 0, 1, 1.0, 0.0, 0.0,
            -4, -12.0, -0.75, 0, 1, 1.0, 0.0, 0.0,
            -4, -9.0, -0.75, 0, 1, 1.0, 0.0, 0.0, # see if this loads.

            # a4 - wrote this way to transition into this.
            -4, -7.0, -0.75, 0, 1, 1.0, 0.0, 0.0,
            3, -9.0, -0.75, 0, 1, 1.0, 0.0, 0.0,
            3, -7.0, -0.75, 0, 1, 1.0, 0.0, 0.0,

            # a7 - instead.
            # this should be the only thing required actually
            5, -7.0, -0.75, 0, 1, 1.0, 0.0, 0.0,
            5, -7.0, -0.75, 0, 1, 1.0, 0.0, 0.0,

            5, -2.0, -0.75, 0, 1, 1.0, 0.0, 0.0,
            -4, -7.0, -0.75, 0, 1, 1.0, 0.0, 0.0,
            -4, -2.0, -0.75, 0, 1, 1.0, 0.0, 0.0,

        )

        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vertex_count = 31

        texture_path = 's64_jimen.png'
        self.texture = self.load_texture(texture_path)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))

    def load_texture(self, path):
        image = Image.open(path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = np.array(list(image.getdata()), np.uint8)

        # Change alpha value
        alpha = 1  # Example: Set alpha to 0.4 for 40% transparency
        alpha = np.clip(alpha, 0.0, 1.0)  # Ensure alpha is within range [0.0, 1.0]
        img_data[:, 3] = (alpha * 255).astype(np.uint8)  # Normalize and convert to uint8

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        return texture

    def destroy(self):

        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))
        glDeleteTextures(1, (self.texture,))


class Quad2D():


    # use generate wall functions here.


    def __init__(self, center, size):


        vertices = (
            20, 6, 1, 0.0, 0, 0.0, 0.0, 0.0,
            20, 6, 5.75, 1.0, 0, 1.0, 0.0, 0.0,
            20, -16, 1, 0, 1, 0.0, 0.0, 0.0,
            20, -16, 5.75, 1, 1, 0.0, 0.0, 0.0,

            -10, -16, 1, 0, 0, 0.0, 0.0, 0.0,
            -10, -16, 5.75, 1, 0, 0.0, 0.0, 0.0,

           -10, 6, 1, 0, 1.0, 0.0, 0.0, 0.0,
           -10, 6, 2.75, 1.0, 1.0, 0.0, 0.0, 0.0,

            20, 6, 1, 0, 0, 0.0, 0.0, 0.0,
            20, 6, 5.75, 1.0, 0, 0.0, 0.0, 0.0,


        )


        self.vertex_count = 10
        self.vertices = np.array(vertices, dtype=np.float32)

        texture_path = 's64_sky.png'
        self.texture = self.load_texture(texture_path)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))



    def load_texture(self, path):
        image = Image.open(path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = np.array(list(image.getdata()), np.uint8)

        # Change alpha value
        alpha = 1  # Example: Set alpha to 0.4 for 40% transparency
        alpha = np.clip(alpha, 0.0, 1.0)  # Ensure alpha is within range [0.0, 1.0]
        img_data[:, 3] = (alpha * 255).astype(np.uint8)  # Normalize and convert to uint8

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        return texture

    def destroy(self):

        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1,(self.vbo,))
        glDeleteTextures(1, (self.texture,))

class ItemMesh():

    def __init__(self):


        # x, y, z, s, t, nx, ny, nz
        self.vertices = load_model_from_file("final_cube_small2.obj")
        self.vertex_count = len(self.vertices) // 8
        self.vertices = np.array(self.vertices, dtype=np.float32)

        texture_path = 'item_text.jpeg'
        self.texture = self.load_texture(texture_path)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)


        #position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        #texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        #normal
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))
        self.load_materials("final_cube_small2.mtl")



    def load_texture(self, path):
        image = Image.open(path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = np.array(list(image.getdata()), np.uint8)



        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        return texture

    def load_materials(self, material_file):
        materials = {}
        current_material = None

        with open(material_file, 'r') as f:
            for line in f:
                line = line.strip()

                if line.startswith('newmtl'):
                    material_name = line.split()[1]
                    current_material = {}
                    materials[material_name] = current_material
                elif current_material is not None:
                    if line.startswith('Ka'):
                        # Ambient color
                        values = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
                        current_material['ambient_color'] = values[:3]
                    elif line.startswith('Kd'):
                        # Diffuse color
                        values = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
                        current_material['diffuse_color'] = values[:3]
                    elif line.startswith('Ks'):
                        # Specular color
                        values = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
                        current_material['specular_color'] = values[:3]
                    elif line.startswith('Ns'):
                        # Shininess
                        shininess = float(line.split()[1])
                        current_material['shininess'] = shininess
                    elif line.startswith('map_Kd'):
                        # Diffuse texture
                        texture_file = line.split()[1]
                        image = Image.open(texture_file)
                        image_data = np.array(image)
                        texture_id = glGenTextures(1)
                        glBindTexture(GL_TEXTURE_2D, texture_id)
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE,
                                     image_data)
                        glGenerateMipmap(GL_TEXTURE_2D)
                        current_material['diffuse_texture'] = texture_id

        self.materials = materials
        self.material_name = material_name

    def render(self, shader_program, material_name):
        #material = self.material_name
        material = self.materials['Material']
        if material:
            # Set material properties in the shader program
            # (Assuming you have a shader program set up)
            print(material)
            glUniform3fv(glGetUniformLocation(shader_program, 'materialDiffuse'), 1, material['diffuse_color'])
            glUniform3fv(glGetUniformLocation(shader_program, 'materialSpecular'), 1, material['specular_color'])

            # Render the mesh
            glBindVertexArray(self.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
            glBindVertexArray(0)

class ShellMesh():

    def __init__(self):

        self.vertices = load_model_from_file("shell.obj")
        self.vertex_count = len(self.vertices) // 8
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)


        #position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        #texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        #normal
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))
        self.load_materials("shell.mtl")


    def load_materials(self, material_file):
        materials = {}
        current_material = None

        with open(material_file, 'r') as f:
            for line in f:
                line = line.strip()

                if line.startswith('newmtl'):
                    material_name = line.split()[1]
                    current_material = {}
                    materials[material_name] = current_material
                elif current_material is not None:
                    if line.startswith('Ka'):
                        # Ambient color
                        values = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
                        current_material['ambient_color'] = values[:3]
                    elif line.startswith('Kd'):
                        # Diffuse color
                        values = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
                        current_material['diffuse_color'] = values[:3]
                    elif line.startswith('Ks'):
                        # Specular color
                        values = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
                        current_material['specular_color'] = values[:3]
                    elif line.startswith('Ns'):
                        # Shininess
                        shininess = float(line.split()[1])
                        current_material['shininess'] = shininess
                    elif line.startswith('map_Kd'):
                        # Diffuse texture
                        texture_file = line.split()[1]
                        image = Image.open(texture_file)
                        image_data = np.array(image)
                        texture_id = glGenTextures(1)
                        glBindTexture(GL_TEXTURE_2D, texture_id)
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE,
                                     image_data)
                        glGenerateMipmap(GL_TEXTURE_2D)
                        current_material['diffuse_texture'] = texture_id

        self.materials = materials
        self.material_name = material_name

    def render(self, shader):
        for material_name, material in self.materials.items():
            # Set material properties in the shader
            glUniform3fv(glGetUniformLocation(shader, 'materialDiffuse'), 1, material['diffuse_color'])
            glUniform3fv(glGetUniformLocation(shader, 'materialSpecular'), 1, material['specular_color'])
            glUniform3fv(glGetUniformLocation(shader, 'materialAmbient'), 1, material['ambient_color'])

            # Render the mesh
            glBindVertexArray(self.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
            glBindVertexArray(0)

class PlayerMesh():

    def __init__(self):


        # x, y, z, s, t, nx, ny, nz
        self.vertices = load_model_from_file("wheel.obj")
        self.vertex_count = len(self.vertices) // 8
        self.vertices = np.array(self.vertices, dtype=np.float32)

        #texture_path = 'item_text.jpeg'
        #self.texture = self.load_texture(texture_path)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)


        #position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        #texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        #normal
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))
        self.load_materials("wheel.mtl")



    def load_texture(self, path):
        image = Image.open(path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = np.array(list(image.getdata()), np.uint8)



        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        return texture

    def load_materials(self, material_file):
        materials = {}
        current_material = None

        with open(material_file, 'r') as f:
            for line in f:
                line = line.strip()

                if line.startswith('newmtl'):
                    material_name = line.split()[1]
                    current_material = {}
                    materials[material_name] = current_material
                elif current_material is not None:
                    if line.startswith('Ka'):
                        # Ambient color
                        values = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
                        current_material['ambient_color'] = values[:3]
                    elif line.startswith('Kd'):
                        # Diffuse color
                        values = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
                        current_material['diffuse_color'] = values[:3]
                    elif line.startswith('Ks'):
                        # Specular color
                        values = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", line)))
                        current_material['specular_color'] = values[:3]
                    elif line.startswith('Ns'):
                        # Shininess
                        shininess = float(line.split()[1])
                        current_material['shininess'] = shininess
                    elif line.startswith('map_Kd'):
                        # Diffuse texture
                        texture_file = line.split()[1]
                        image = Image.open(texture_file)
                        image_data = np.array(image)
                        texture_id = glGenTextures(1)
                        glBindTexture(GL_TEXTURE_2D, texture_id)
                        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE,
                                     image_data)
                        glGenerateMipmap(GL_TEXTURE_2D)
                        current_material['diffuse_texture'] = texture_id

        self.materials = materials
        self.material_name = material_name

    def render(self, shader):
        for material_name, material in self.materials.items():
            # Set material properties in the shader
            glUniform3fv(glGetUniformLocation(shader, 'materialDiffuse'), 1, material['diffuse_color'])
            glUniform3fv(glGetUniformLocation(shader, 'materialSpecular'), 1, material['specular_color'])
            glUniform3fv(glGetUniformLocation(shader, 'materialAmbient'), 1, material['ambient_color'])

            # Render the mesh
            glBindVertexArray(self.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
            glBindVertexArray(0)



class WallMesh():

    # generate walls up the top.
    def generate_vertical_wall(self, start, finish, y_level, cliff):

        # generate a random number and put them in.
        # return a list of walls that can be appended onto veritcies
        # generate just a standard wall.
        size = 0
        if not cliff:
            vertices = (  # simple flat wall
                start, y_level, 0, 0, 0, 1.0, 0.0, 0.0,  # bottom left vertex
                finish, y_level, 0, 1, 0, 1.0, 0.0, 0.0,  # bottom right
                start, y_level, 1, 0, 1, 0.0, 1.0, 0.0,  # top left

                finish, y_level, 0, 1, 0, 1.0, 0.0, 0.0,  # bottom right vertex
                start, y_level, 1, 0, 1, 0.0, 1.0, 0.0,  # top left vertex
                finish, y_level, 1, 1, 1, 0.0, 1.0, 0.0,)  # top right vertex )

            size = 6
        elif cliff:
            # generate random number between 1 and 3 about how far the protrusion should be
            # so here -> change them up.
            p_level = random.randint(1, 5)

            if y_level < 0:
                # change the direction that the cliff faces, needs to face into the thing.
                vertices = (
                    start, y_level, -0.75, 0, 0, 1.0, 1.0, 1.0,  #
                    start, y_level, 1.75, 1, 0, 1.0, 1.0, 1.0,  # v3
                    start + p_level, y_level, 1.75, 1, 1, 1.0, 1.0, 1.0,  # v3

                    start, y_level, -0.75, 0, 1, 1.0, 1.0, 1.0,  # v1

                    # stopped here.
                    start, y_level + 1, -0.75, 0, 0, 1.0, 1.0, 1.0,  # v4
                    start + (p_level + 0.5), y_level + 1, -0.75, 0, 1, 1.0, 1.0, 1.0,  # v5

                    start + p_level, y_level, 1.75, 1, 0, 0.0, 0.0, 0.0,  # v3 = v
                    finish, y_level, -0.75, 1, 1, 0.0, 0.0, 0.0,  # v8
                    finish, y_level, 1.75, 1, 0, 0.0, 0.0, 0.0,)  # v7)

            else:

                vertices = (
                    start, y_level, -0.75, 0, 0, 0.0, 1.0, 0.0,  # v3
                    start, y_level, 1.75, 1, 0, 0.0, 1.0, 0.0,  # v3
                    start + p_level, y_level, 1.75, 1, 1, 0.0, 1.0, 0.0,  # v3

                    start, y_level, -0.75, 0, 0, 1.0, 0.0, 0.0,  # v1
                    start, y_level - 1, -0.75, 0, 0, 1.0, 0.0, 0.0,  # v4
                    start + (p_level + 0.5), y_level - 1, -0.75, 1, 0, 0.0, 1.0, 0.0,  # v5

                    start + p_level, y_level, 1.75, 0, 0, 0.0, 1.0, 0.0,  # v3 = v
                    finish, y_level, -0.75, 1, 1, 1.0, 0.0, 0.0,  # v8
                    finish, y_level, 1.75, 1, 0, 0.0, 1.0, 0.0,)  # v7)





            size = 9
        return size, vertices

    # put these functions in the appendix and see how you go.
    def generate_horizontal_wall(self, start, finish, x_level, cliff):
        # generate a random number and put them in.
        # return a list of walls that can be appended onto veritcies
        # generate just a standard wall.
        size = 0
        if not cliff:
            vertices = (  # simple flat wall
                x_level, start, -0.75, 0, 0, 1.0, 0.0, 0.0,  # bottom left vertex
                x_level, finish, -0.75, 0, 1, 1.0, 0.0, 0.0,  # bottom right
                x_level, start, 1.75, 1, 0, 0.0, 1.0, 0.0,  # top left

                x_level, finish, -0.75, 0, 0, 1.0, 0.0, 0.0,  # bottom right vertex
                x_level, start, 1.75, 1, 0, 0.0, 1.0, 0.0,  # top left vertex
                x_level, finish, 1.75, 1, 1, 0.0, 1.0, 0.0,)  # top right vertex )

            size = 6

        return size, vertices

    def __init__(self):
        # generate the walls and see how you go.

        self.vertices = (

            # okay - so it in little time steps at a time.

            16, 2, 0, 0, 0, 1.0, 0.0, 0.0,  # bottom left vertex
            16, -2, 0, 1, 0, 1.0, 0.0, 0.0,  # bottom right
            16, 2, 1, 0, 1, 0.0, 1.0, 0.0,  # top left

            16, -2, 0, 1, 0, 1.0, 0.0, 0.0,  # bottom right vertex
            16, 2, 1, 0, 1, 0.0, 1.0, 0.0,  # top left vertex
            16, -2, 1, 1, 1, 0.0, 1.0, 0.0,

        )

        # generate on 16 x 14 grid

        self.vertex_count = 0

        # on the side y = 2
        l0, v0 = self.generate_vertical_wall(-4, 0, 2, True)
        l1, v1 = self.generate_vertical_wall(0, 4, 2, True)
        l2, v2 = self.generate_vertical_wall(5, 9, 2, True)
        l3, v3 = self.generate_vertical_wall(10, 13, 2, True)
        l4, v4 = self.generate_vertical_wall(14, 16, 2, True)

        # generate middle horizontal wall.
        # x level = 16, until -12
        lh1, vh1 = self.generate_horizontal_wall(2, -2, 16, False)
        lh2, vh2 = self.generate_horizontal_wall(-2, -5, 16, False)
        lh3, vh3 = self.generate_horizontal_wall(-5, -9, 16, False)
        lh4, vh4 = self.generate_horizontal_wall(-9, -12, 16, False)

        # on other side y = - 12a
        l12, v12 = self.generate_vertical_wall(-4, 4, -12, True)
        l22, v22 = self.generate_vertical_wall(5, 9, -12, True)
        l32, v32 = self.generate_vertical_wall(10, 13, -12, True)
        l42, v42 = self.generate_vertical_wall(14, 16, -12, True)

        # other way horizontal walls
        # x = 0
        lh11, vh11 = self.generate_horizontal_wall(-12, -9, -4, False)
        lh22, vh22 = self.generate_horizontal_wall(-9, -5, -4, False)
        lh33, vh33 = self.generate_horizontal_wall(-5, -1, -4, False)
        lh44, vh44 = self.generate_horizontal_wall(-1, 2, -4, False)


        self.vertices = v0 + v1 + v2 + v3 + v4 + vh1 + vh2 + vh3 + vh4 + v42 + v32 + v22 + v12 + vh11+ vh22+ vh33+ vh44
        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vertex_count = l0 + l1 + l2 + l3 + l4 + l12+ l22+ l32+ l42 + lh1 + lh2 + lh3 + lh4 + lh44*4

        texture_path = 's64_kabe01.png'
        self.texture = self.load_texture(texture_path)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))

        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))

        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))

    def load_texture(self, path):
        image = Image.open(path)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        img_data = np.array(list(image.getdata()), np.uint8)

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        glGenerateMipmap(GL_TEXTURE_2D)

        return texture

    def destroy(self):
        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vbo,))
        glDeleteTextures(1, (self.texture,))


myApp = App()
