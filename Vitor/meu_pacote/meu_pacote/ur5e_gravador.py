from controller import Robot, Keyboard

# --- CONFIGURAÇÕES ---
TIME_STEP = 32
VEL_MANUAL = 1.0       # Velocidade ao mover com teclas
VEL_PLAYBACK = 1.5     # Velocidade ao reproduzir o movimento
TOLERANCIA = 0.05      # Precisão (radianos)
DELAY_CLIQUE = 0.5     # [NOVO] Tempo de espera (segundos) para botões de comando

class UR5eFinal:
    def __init__(self):
        self.robot = Robot()
        self.keyboard = Keyboard()
        self.keyboard.enable(TIME_STEP)

        # Configuração dos Motores
        self.joint_names = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        
        self.motors = []
        self.sensors = []

        print("--> Configurando Hardware...")
        for name in self.joint_names:
            m = self.robot.getDevice(name)
            m.setPosition(float('inf'))
            m.setVelocity(0.0)
            self.motors.append(m)
            
            s = self.robot.getDevice(name + '_sensor')
            if s is None: s = m.getPositionSensor()
            if s:
                s.enable(TIME_STEP)
                self.sensors.append(s)

        # Variáveis de Estado
        self.poses = []
        self.estado = 'MANUAL'
        self.ultimo_tempo_cmd = 0.0  # Marca quando foi o último clique de comando

        print("=== SISTEMA PRONTO (COM DELAY) ===")
        print(" [WASD/QE] Mover Braço")
        print(" [P]       Mudar Modo (Manual <-> Gravação)")
        print(" [O]       Salvar Pose")
        print(" [F]       Finalizar e Tocar")

    def run(self):
        while self.robot.step(TIME_STEP) != -1:
            key = self.keyboard.getKey()
            agora = self.robot.getTime() # Tempo atual da simulação
            
            # --- COMANDO P: MUDANÇA DE MODO ---
            if key == ord('P'):
                # Só aceita se passou o tempo do delay
                if agora - self.ultimo_tempo_cmd > DELAY_CLIQUE:
                    self.alternar_modo()
                    self.ultimo_tempo_cmd = agora # Reseta o relógio do delay

            # --- LÓGICA DOS ESTADOS ---
            if self.estado == 'MANUAL':
                self.control_manual(key)

            elif self.estado == 'GRAVANDO':
                self.logic_gravacao(key, agora)

            elif self.estado == 'REPRODUZINDO':
                pass 

    def alternar_modo(self):
        if self.estado == 'MANUAL':
            self.estado = 'GRAVANDO'
            self.poses = []
            print(f"\n>>> MODO GRAVAÇÃO <<<")
            # Para o robô
            for m in self.motors: m.setVelocity(0.0)
        
        elif self.estado == 'GRAVANDO' or self.estado == 'REPRODUZINDO':
            self.estado = 'MANUAL'
            print(f"\n<<< MODO MANUAL <<<")
            for m in self.motors: 
                m.setPosition(float('inf'))
                m.setVelocity(0.0)

    def logic_gravacao(self, key, agora):
        # Movimentação (Sem delay, fluida)
        self.control_manual(key)

        # COMANDO O: SALVAR POSE
        if key == ord('O'):
            if agora - self.ultimo_tempo_cmd > DELAY_CLIQUE:
                pose_atual = [s.getValue() for s in self.sensors]
                self.poses.append(pose_atual)
                print(f" [+] Pose {len(self.poses)} Salva!")
                self.ultimo_tempo_cmd = agora

        # COMANDO F: FINALIZAR
        elif key == ord('F'):
            if agora - self.ultimo_tempo_cmd > DELAY_CLIQUE:
                self.ultimo_tempo_cmd = agora
                
                if len(self.poses) == 0:
                    print(" [!] Nada gravado para tocar.")
                else:
                    self.estado = 'REPRODUZINDO'
                    print(f"\n>>> REPRODUZINDO {len(self.poses)} POSES... <<<")
                    self.executar_playback()
                    
                    # Ao terminar, volta para manual
                    self.estado = 'MANUAL'
                    print(">>> Fim. Voltando para Manual. <<<")
                    for m in self.motors:
                        m.setPosition(float('inf'))
                        m.setVelocity(0.0)

    def executar_playback(self):
        for i, pose in enumerate(self.poses):
            print(f" -> Pose {i+1}...", end=" ")
            
            # Envia motores
            for j, m in enumerate(self.motors):
                m.setVelocity(VEL_PLAYBACK)
                m.setPosition(pose[j])
            
            # Espera chegar
            chegou = False
            t_inicio = self.robot.getTime()
            
            while not chegou:
                if self.robot.step(TIME_STEP) == -1: return

                erro_max = 0
                for j, s in enumerate(self.sensors):
                    diff = abs(s.getValue() - pose[j])
                    if diff > erro_max: erro_max = diff
                
                if erro_max < TOLERANCIA:
                    chegou = True
                    print("OK")
                
                # Timeout de segurança (4s)
                if self.robot.getTime() - t_inicio > 4.0:
                    print("(Timeout)")
                    chegou = True

    def control_manual(self, key):
        vels = [0.0] * 6
        v = VEL_MANUAL
        
        if key == ord('A'): vels[0] = v
        elif key == ord('D'): vels[0] = -v
        elif key == ord('W'): vels[1] = v
        elif key == ord('S'): vels[1] = -v
        elif key == ord('Q'): vels[2] = v
        elif key == ord('E'): vels[2] = -v
        
        for i, m in enumerate(self.motors):
            m.setVelocity(vels[i])
            if self.estado != 'REPRODUZINDO':
                m.setPosition(float('inf'))

if __name__ == "__main__":
    app = UR5eFinal()
    app.run()