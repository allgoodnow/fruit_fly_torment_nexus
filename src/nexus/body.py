"""Thin adapter around the official FlyGym hybrid turning example.

The controller is engineered locomotion, NOT a fruit fly connectome.
All rendering and physics objects are owned by a single worker process/thread.
"""
from __future__ import annotations

import math
from collections import deque

import mujoco
import numpy as np
from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld, ActuatorType
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridControllerObservation, HybridTurningController, LocomotionAction,
    PreprogrammedSteps, apply_locomotion_action, make_locomotion_fly,
)


class FlyBody:
    def __init__(self, *, render: bool = True, width: int = 960, height: int = 640):
        self.fly = make_locomotion_fly(name="nexus_fly", add_adhesion=True, colorize=True)
        self.world = FlatGroundWorld()
        for texture in self.world.mjcf_root.textures:
            if texture.name == 'checker':
                texture.rgb1 = [.92, .93, .94]
                texture.rgb2 = [.985, .985, .985]
        for material in self.world.mjcf_root.materials:
            if material.name == 'grid':
                material.reflectance = 0
        self.world.add_fly(
            self.fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]),
            bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
            add_ground_contact_sensors=False,
        )
        self.sim = Simulation(self.world)
        self.steps = PreprogrammedSteps()
        self.dofs = self.fly.get_actuated_jointdofs_order("position")
        self.controller = HybridTurningController(
            timestep=self.sim.timestep, preprogrammed_steps=self.steps,
            output_dof_order=self.dofs,
        )
        self.thorax_index = self.fly.get_bodysegs_order().index(BodySegment("c_thorax"))
        self.thorax_id = self.sim._internal_bodyids_by_fly[self.fly.name][self.thorax_index]
        model = self.sim.mj_model
        actuator_ids = self.sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION][self.fly.name]
        joint_ids = model.actuator_trnid[actuator_ids, 0]
        self.angle_low = np.where(model.jnt_limited[joint_ids], model.jnt_range[joint_ids, 0], -np.inf)
        self.angle_high = np.where(model.jnt_limited[joint_ids], model.jnt_range[joint_ids, 1], np.inf)
        self.effect_phase = np.arange(len(self.dofs))*1.61803398875
        self.effect_frequency = 9.+(np.arange(len(self.dofs)) % 7)*1.7
        self.ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'ground_plane')
        self.foot_geoms = {i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i).split('/')[-1][:2]
                           for i in range(model.ngeom)
                           if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or '').endswith('tarsus5')}
        if self.ground_id < 0 or len(self.foot_geoms) != 6:
            raise RuntimeError('Expected ground plane and six distal tarsus contact geometries')
        self.food_patch = None
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.reset_camera()
        self.renderer = None
        self.width, self.height = width, height
        self.elapsed_steps = 0
        self.path = deque(maxlen=1200)
        self.reset()
        if render:
            self.sim.mj_model.vis.global_.offwidth = max(1024, width)
            self.sim.mj_model.vis.global_.offheight = max(768, height)
            self.renderer = mujoco.Renderer(self.sim.mj_model, height=height, width=width)

    @property
    def time(self):
        return self.elapsed_steps * self.sim.timestep

    def reset_camera(self):
        self.camera.azimuth = 135
        self.camera.elevation = -25
        self.camera.distance = 7.0

    def reset(self):
        self.sim.reset()
        self.controller.reset(seed=0)
        apply_locomotion_action(self.sim, self.fly.name, LocomotionAction(
            joint_angles=self.steps.default_pose_by_dof_order(self.dofs),
            adhesion_onoff=np.ones(6, dtype=bool),
        ))
        self.sim.warmup()
        self.elapsed_steps = 0
        self.path.clear()
        self.origin = self.position().copy()
        self.motor_offset_rms = 0.
        self.rest_angles = None

    def position(self):
        return self.sim.mj_data.xpos[self.thorax_id].copy()

    def upright(self):
        return float(self.sim.mj_data.xmat[self.thorax_id].reshape(3, 3)[2, 2])

    def reposition(self):
        """Explicit posture assistance; preserve location, clock and trail."""
        free = np.flatnonzero(self.sim.mj_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1:
            raise RuntimeError('Body reposition requires one free root joint')
        address = int(self.sim.mj_model.jnt_qposadr[free[0]])
        xy, steps, physics_time = self.position()[:2], self.elapsed_steps, self.sim.mj_data.time
        origin, path = self.origin.copy(), list(self.path)
        self.reset()
        mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)
        self.sim.mj_data.qpos[address:address+2] += xy-self.position()[:2]
        self.sim.mj_data.qvel[:] = 0
        self.sim.mj_data.time = physics_time
        mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)
        self.elapsed_steps, self.origin = steps, origin
        self.path.extend(path)

    def ground_contacts(self):
        result = []
        for contact in self.sim.mj_data.contact[:self.sim.mj_data.ncon]:
            other = int(contact.geom2) if contact.geom1 == self.ground_id else int(contact.geom1) if contact.geom2 == self.ground_id else -1
            if other in self.foot_geoms and contact.dist <= 0 and contact.efc_address >= 0:
                result.append({'foot': self.foot_geoms[other], 'position_mm': contact.pos.copy().tolist()})
        return result

    def food_position(self, placement):
        position = self.position()[:2]
        if placement == 'ahead':
            forward = self.sim.mj_data.xmat[self.thorax_id].reshape(3, 3)[:2, 0]
            return (position+forward/max(np.linalg.norm(forward), 1e-9)*5).tolist()
        if placement == 'under':
            return position.tolist()
        raise ValueError('Choose food placement ahead or under')

    def advance(self, seconds: float, *, drive: float = 1.0, turn: float = 0.0,
                wander: bool = True, escape: float = 0., disruption: float = 0., resting: bool = False):
        if not all(math.isfinite(float(v)) and 0 <= v <= 1 for v in (escape, disruption)):
            raise ValueError('Motor effects must be finite levels between zero and one')
        count = max(1, round(seconds / self.sim.timestep))
        drive = float(np.clip(drive+.3*escape, -.8, 1.3))
        turn = float(np.clip(turn, -0.6, 0.6))
        for _ in range(count):
            # Authored exploration signal. This is deliberately not called a brain.
            steering = (0.22 * math.sin(self.time * 0.8) + 0.1 * math.sin(self.time * 2.1)) if wander else turn
            # Preserve the original nonnegative commands during forward walking.
            # Signed upstream CPG frequencies reverse the step cycle for retreat.
            signal = np.clip([drive + steering, drive - steering],
                             -.8 if drive < 0 else 0., 0. if drive < 0 else 1.5)
            obs = HybridControllerObservation.from_sim(self.sim, self.fly.name)
            if resting and not (escape or disruption):
                if self.rest_angles is None:
                    self.rest_angles = self.steps.default_pose_by_dof_order(self.dofs).copy()
                action = LocomotionAction(joint_angles=self.rest_angles, adhesion_onoff=np.ones(6, dtype=bool))
            else:
                self.rest_angles = None
                action = self.controller.step(signal, obs)
            if escape or disruption:
                # Authored muscle-command disturbance driven by neural readouts.
                # No pose teleport, injected body force, or thermal tissue model.
                phase = 2*np.pi*self.time
                offset = (.18*escape*np.sin(phase*5+self.effect_phase)
                          + .8*disruption*np.sin(phase*self.effect_frequency+self.effect_phase))
                angles = np.clip(action.joint_angles+offset, self.angle_low, self.angle_high)
                adhesion = action.adhesion_onoff.copy()
                adhesion &= np.sin(phase*12+np.arange(6)*2.4) < 1-1.8*disruption
                self.motor_offset_rms = float(np.sqrt(np.mean((angles-action.joint_angles)**2)))
                action = LocomotionAction(joint_angles=angles, adhesion_onoff=adhesion)
            else:
                self.motor_offset_rms = 0.
            apply_locomotion_action(self.sim, self.fly.name, action)
            self.sim.step()
            self.elapsed_steps += 1
        if not np.isfinite(self.sim.mj_data.qpos).all() or not np.isfinite(self.sim.mj_data.qvel).all():
            raise RuntimeError("Physics produced a non-finite state; simulation stopped.")
        self.path.append(self.position()[:2].tolist())

    def orbit(self, dx: float = 0, dy: float = 0, zoom: float = 0):
        self.camera.azimuth = (self.camera.azimuth + dx) % 360
        self.camera.elevation = float(np.clip(self.camera.elevation + dy, -89, -3))
        self.camera.distance = float(np.clip(self.camera.distance * math.exp(zoom), 2.5, 35))

    def render(self):
        if self.renderer is None:
            raise RuntimeError("Renderer was not initialized")
        self.camera.lookat[:] = self.position()
        self.renderer.update_scene(self.sim.mj_data, camera=self.camera)
        if self.food_patch and self.food_patch['present']:
            scene = self.renderer.scene
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError('No render geometry slot available for food')
            patch = self.food_patch
            # A visual ground annotation, not an extra collidable object. Taste
            # uses the same centre/radius with actual foot/ground contacts.
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                               np.array([patch['radius_mm'], .004, 0.]),
                               np.array([*patch['center_mm'], .004]), np.eye(3).ravel(),
                               np.array([.82, .63, .19, .85], dtype=np.float32))
            scene.ngeom += 1
        return self.renderer.render().copy()

    def telemetry(self):
        return {
            "sim_time": self.time,
            "position_mm": self.position().tolist(),
            "displacement_mm": float(np.linalg.norm(self.position()[:2] - self.origin[:2])),
            "contacts": int(self.sim.mj_data.ncon),
            "joints": int(self.sim.mj_model.njnt),
            "actuators": int(self.sim.mj_model.nu),
            "phases": (self.controller.cpg_network.curr_phases % (2 * np.pi)).tolist(),
            "magnitudes": self.controller.cpg_network.curr_magnitudes.tolist(),
            "path": list(self.path),
            "motor_offset_rms_rad": self.motor_offset_rms,
            "upright": self.upright(),
            "food_patch_present": bool(self.food_patch and self.food_patch['present']),
            "mode": "engineered locomotion; connectome not connected",
        }

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
